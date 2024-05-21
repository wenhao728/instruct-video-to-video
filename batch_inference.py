#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import time
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.single_video_dataset import SingleVideoDataset
from insv2v_run_loveu_tgve import split_batch
from misc_utils.train_utils import unit_test_create_model
from pl_trainer.inference.inference import InferenceIP2PVideoOpticalFlow
from save_video import save_video


POS_PROMPT = (
    " ,best quality, extremely detailed, HD, ultra, 8K, HQ, masterpiece, trending on artstation, art, smooth")
NEG_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, deformed body, bloated, ugly, blurry, low res, unaesthetic"
)

data_root = '/workspace/DynEdit'
method_name = 'insv2v'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config_1.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=33,
    # TODO define arguments
    config_path='/workspace/instruct-video-to-video/configs/instruct_v2v.yaml',
    ckpt_path='/workspace/models/insv2v.pth',
    text_cfg=7.5,
    video_cfg=1.8,
    num_frames=24,
    image_size=512,
    num_ddim_steps=20,
    frames_in_batch=16,
    num_ref_frames=4,
    fps=12,
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    diffusion_model = unit_test_create_model(config.config_path)
    ckpt = torch.load(config.ckpt_path, map_location='cpu')
    ckpt = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
    diffusion_model.load_state_dict(ckpt, strict=False)
    inf_pipe = InferenceIP2PVideoOpticalFlow(
        unet=diffusion_model.unet,
        num_ddim_steps=config.num_ddim_steps,
        scheduler='ddpm'
    )

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/videos/{row.video_id}.mp4'
        # TODO load video
        dataset = SingleVideoDataset(
            video_path, video_description=row.prompt, 
            sampling_fps=12, num_frames=24, output_size=(512, 512), 
        )

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        batch = {
            k: v.cuda()[None] 
            if isinstance(v, torch.Tensor) else v 
            for k, v in dataset[0].items()
        }
        cond = diffusion_model.encode_image_to_latent(batch['frames']) / 0.18215
        text_uncond = diffusion_model.encode_text([''])
        conds, num_ref_frames_each_batch = split_batch(
            cond, frames_in_batch=config.frames_in_batch, num_ref_frames=config.num_ref_frames)
        splitted_frames, _ = split_batch(
            batch['frames'], frames_in_batch=config.frames_in_batch, num_ref_frames=config.num_ref_frames)

        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            
            text_cond = diffusion_model.encode_text(f'change {edit.src_words} to {edit.tgt_words}')
            # First video clip
            cond1 = conds[0]
            latent_pred_list = []
            init_latent = torch.randn_like(cond1)
            latent_pred = inf_pipe(
                latent = init_latent,
                text_cond = text_cond,
                text_uncond = text_uncond,
                img_cond = cond1,
                text_cfg = config.text_cfg,
                img_cfg = config.video_cfg,
            )['latent']
            latent_pred_list.append(latent_pred)
            # Subsequent video clips
            for prev_cond, cond_, prev_frame, curr_frame, num_ref_frames_ in zip(
                conds[:-1], conds[1:], splitted_frames[:-1], splitted_frames[1:], num_ref_frames_each_batch):

                init_latent = torch.cat([init_latent[:, -num_ref_frames_:], torch.randn_like(cond_)], dim=1)
                cond_ = torch.cat([prev_cond[:, -num_ref_frames_:], cond_], dim=1)
                ref_images = prev_frame[:, -num_ref_frames_:]
                query_images = curr_frame
                additional_kwargs = {
                    'ref_images': ref_images,
                    'query_images': query_images,
                }
                latent_pred = inf_pipe.second_clip_forward(
                    latent = init_latent, 
                    text_cond = text_cond,
                    text_uncond = text_uncond,
                    img_cond = cond_,
                    latent_ref = latent_pred[:, -num_ref_frames_:],
                    noise_correct_step = 0.5,
                    text_cfg = config.text_cfg,
                    img_cfg = config.video_cfg,
                    **additional_kwargs,
                )['latent']
                latent_pred_list.append(latent_pred[:, num_ref_frames_:])
        
        latent_pred = torch.cat(latent_pred_list, dim=1)
        image_pred = diffusion_model.decode_latent_to_image(latent_pred).clip(-1, 1)
        transferred_images = image_pred.squeeze(0).detach().float().cpu().numpy() / 2 + 0.5
        save_video(f'{output_dir}/{i}.mp4', transferred_images, fps=config.fps)

        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()