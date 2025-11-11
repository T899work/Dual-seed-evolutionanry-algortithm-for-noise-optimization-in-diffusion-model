import os
import torch
from open_clip import image_transform
import random
import json
from transformers import CLIPModel
import ImageReward as RM
from transformers import AutoProcessor, AutoModel, CLIPProcessor
import clip
from aesthetics_predictor import AestheticsPredictorV2Linear

from pipeline_stable_diffusion_2_1_seed_opti import StableDiffusionPipeline
from diffusers import DDIMScheduler, DDIMInverseScheduler
import numpy as np
from encode_images import vae_encode, clip_encode
from norm_aware_optimization import norm_aware_interpolation
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="args for Z-sampling")
    parser.add_argument('--gamma_1', type=float, default=7.5, help='guidance for denoising process')
    parser.add_argument('--gamma_2', type=float, default=0, help='guidance for inversion process')
    parser.add_argument('--infer_step', type=int, default=50, help='total inference timestep T')
    parser.add_argument('--image_size', type=int, default=512, help='The size (height and width) of the generated image.')
    parser.add_argument('--seed', type=int, default=44, help='Random seed to determine the initial latent.')
    parser.add_argument('--device', type=str, default='cuda', help='Device where the model inference is performed.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of generate image per prompt')
    parser.add_argument('--Num_epoch', type=int, default=3, help='Number of optimization epoch')
    parser.add_argument('--prompt', default='An athletic middle aged male skier courses downhill.', type=str)

    args = parser.parse_args()
    return args

def load_prompts(dataset_choice):
    if dataset_choice == "pickapic":
        json_path = '~'
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]
        token_indices = [item["token_indices"] for item in data if "token_indices" in item]

    elif dataset_choice == "drawbench":
        json_path = "~"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]
        token_indices = [item["token_indices"] for item in data if "token_indices" in item]

    elif dataset_choice == "hpdv2":
        json_path = "~"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item["prompt"] for item in data if "prompt" in item]
        token_indices = [item["token_indices"] for item in data if "token_indices" in item]

    return prompts, token_indices

def calc_pickscore(prompt, images, Pick_model, processor, device='cuda'):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = Pick_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = Pick_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = Pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

    return scores.cpu().tolist()
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):

    if not isinstance(v0, np.ndarray):
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2).to("cuda")

    return v2


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def clip_img_transform(feature_extractor):
    image_mean = feature_extractor.image_mean
    image_std = feature_extractor.image_std
    preprocess_val = image_transform(
        feature_extractor.size["shortest_edge"],
        is_train=False,
        mean=image_mean,
        std=image_std
    )
    return preprocess_val


def calculate_clip_score(images, prompts, device="cuda"):
    assert len(images) == len(prompts), "Each image must correspond to a prompt"

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()
    image_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    text_tokens = clip.tokenize(prompts[0]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensors).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features * text_features).sum(dim=-1)
        return similarities.mean().item()

def prepare_models():
    clip = CLIPModel.from_pretrained("~") 
    clip.vision_model.to("cuda")
    clip.visual_projection.to("cuda")
    clip.eval()
    freeze_params(clip.parameters())
    print('already load Clip model')
    model_id = "~"  
    sd_model = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    sd_model.safety_checker = None
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        model_id, subfolder='scheduler'
    )
    sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
    sd_model.inv_scheduler = inverse_scheduler

    clip_transform = clip_img_transform(sd_model.feature_extractor)

    return sd_model, sd_model.vae, clip, clip_transform


def optimize_seed(init_seed, prompt, sd_model,j,q,
                  n_iters=1, guidance_scale=7.5, lr=0.01, show_first_image=True):
    img_seed = torch.nn.Parameter(init_seed.reshape((1, 4, 64, 64)), requires_grad=True)
    img_seed = img_seed.to(torch.float32)
    for i in range(n_iters):
        image_pil = sd_model(prompt=prompt, shape=shape, guidance_scale=guidance_scale, num_inference_steps=args.infer_step, latents=img_seed).images
    return image_pil

def get_init_latents(random_seed_fix, batch_size):

    latents=[]
    for i in range(batch_size):
        random_seed = random.randint(0, 500000000)
        np.random.seed(int(random_seed))
        torch.manual_seed(int(random_seed))
        torch.cuda.manual_seed(int(random_seed))
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        start_latents=torch.randn(*shape[1:], generator=generator).to('cuda')
        latents.append(start_latents)
    return torch.stack(latents)

if __name__ == "__main__":

    args = get_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    img_folder = args.save_dir
    sd_model, vae, clip_model, clip_transform = prepare_models()
    RMmodel = RM.load("ImageReward-v1.0")
    AES_model_id = "~" 
    predictor = AestheticsPredictorV2Linear.from_pretrained(AES_model_id)
    AES_processor = CLIPProcessor.from_pretrained(AES_model_id)
    processor_name_or_path = "~"  
    Pick_model_pretrained_name_or_path = "~" 
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    Pick_model = AutoModel.from_pretrained(Pick_model_pretrained_name_or_path).eval().to('cuda')


    clip_centroid = clip_encode(clip_model, clip_transform, img_folder).mean(dim=0)
    clip_centroid = clip_centroid / clip_centroid.norm(dim=-1, keepdim=True)

    vae_latents = vae_encode(vae, img_folder)
    vae_centroid = vae_latents.mean(dim=0).unsqueeze(0)
    del vae_centroid,vae_latents,clip_centroid, vae, clip_model, clip_transform
    torch.cuda.empty_cache()  

    height = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    width = sd_model.unet.config.sample_size * sd_model.vae_scale_factor
    shape = (1, sd_model.unet.in_channels, height // sd_model.vae_scale_factor, width // sd_model.vae_scale_factor)
    batch_size = args.batch_size
    shape = (batch_size, 4, args.image_size // 8, args.image_size // 8)
    whole_image_clip_score = []
    whole_image_reward_score = []
    std_hpsv2_scores = []
    std_AES_score = []
    whole_pick_score = []
    best_init_latents = []
    for ii in range(0, 1, 1):
            init_latent = get_init_latents(args.seed, batch_size)
            prompt = args.prompt
            print(f'prompt: {prompt}')
            prompts = [prompt] * batch_size
            img = sd_model(prompt=prompts, shape=shape, guidance_scale=args.gamma_1,
                            num_inference_steps=args.infer_step, latents=init_latent).images
            img_paths = []
            all_images = []
            std_clip_score = []
            best_init_latents = []
            for img_idx, image in enumerate(img):
                path = "~" + "image_" + str(
                    img_idx) + ".jpg"
                image.save(path)
                img_paths.append(path)
                all_images.append(image)
            for img_idx, image in enumerate(img):
                std_score = calculate_clip_score([image], [prompt], device='cuda')
                std_clip_score.append(std_score)
                del image, std_score
                torch.cuda.empty_cache()
            ranking, rewards = RMmodel.inference_rank(prompt, img_paths)
            print('rewards', rewards)
            best_index = np.array(ranking).argmin()
            print('Reward_best_index',best_index)
            max_score = max(std_clip_score)
            max_index = std_clip_score.index(max_score)
            max_reward = rewards[best_index]
            max_clip_score = std_clip_score[max_index]
            print('Clip_best_index', max_index)
            for img_idx, image in enumerate(img):
                if img_idx == best_index:
                    path = "~" + "image_" + str(
                        0) + ".jpg"
                    image.save(path)
                    best_reward_image = [image]
                    best_init_latents.append(init_latent[img_idx])

            for img_idx, img in enumerate(img):
                if img_idx == max_index:
                    path = "~" + "image_" + str(
                        1) + ".jpg"
                    best_clip_image = [img]
                    img.save(path)
                    best_init_latents.append(init_latent[img_idx])

            print('std_clip_score', std_clip_score)
            del img,all_images,ranking, rewards
            torch.cuda.empty_cache()
            save_dir = "~"# set path to save
            k = 10
            for op in range(args.Num_epoch-1):
                p1 = best_init_latents[0].reshape((1, 4, 64, 64)).to("cuda")
                p2 = best_init_latents[1].reshape((1, 4, 64, 64)).to("cuda")
                image_best_reward, p1_new = sd_model.few_step_inverse_sampling_call(prompt=prompt, shape=shape,
                                                     guidance_scale=args.gamma_1, inv_guidance_scale=args.gamma_2,
                                                     num_inference_steps=args.infer_step,
                                                     latents=p1, T_max=args.T_max,
                                                     lambda_step=args.lambda_step)
                image_best_reward = sd_model(prompt=prompt, shape=shape, guidance_scale=args.gamma_1,
                               num_inference_steps=args.infer_step, latents=p1_new)

                path = "~" + "image_" + str(
                    0) + ".jpg"
                image_best_reward.images[0].save(path)
                _, rewards = RMmodel.inference_rank(prompt, [path])
                print('current max reward_',max_reward,'_current reward_',rewards)
                if rewards > (max_reward):
                    path = "~" + "image_" + str(
                        0) + ".jpg"
                    image_best_reward.images[0].save(path)
                    print('best image reward update')
                    max_reward=rewards
                    p1 = p1_new
                    best_init_latents[0] = p1_new
                    best_reward_image = image_best_reward.images
                best_clip_image, p2_new = sd_model.few_step_inverse_sampling_call(prompt=prompt, shape=shape,
                                                                             guidance_scale=args.gamma_1,
                                                                             inv_guidance_scale=args.gamma_2,
                                                                             num_inference_steps=args.infer_step,
                                                                             latents=p2, T_max=args.T_max,
                                                                             lambda_step=args.lambda_step)

                best_clip_image = sd_model(prompt=prompt, shape=shape, guidance_scale=args.gamma_1,
                                             num_inference_steps=args.infer_step, latents=p2_new)
                path = "./res_seedselect/best/temp/" + "image_" + str(
                    1) + ".jpg"
                best_clip_image.images[0].save(path)
                std_score = calculate_clip_score(best_clip_image.images, [prompt], device='cuda')
                print('current max Clip_', max_clip_score, '_current reward_', std_score)
                if std_score > (max_clip_score):
                    path = "./res_seedselect/best/" + "image_" + str(
                        1) + ".jpg"
                    best_clip_image.images[0].save(path)
                    print('best image Clip update')
                    max_clip_score = std_score
                    p2 = p2_new
                    best_init_latents[1] = p2_new
                del best_clip_image, image_best_reward
                torch.cuda.empty_cache()
                dim = 4 * 64 * 64
                all_images = []
                n_points = 50
                n_optimization_iters = 1
                lr = 0.01
                p1 = p1.reshape(1, -1).T
                p2 = p2.reshape(1, -1).T
                eps = 2 * (torch.norm(p1 - p2) / n_points)
                log_chi_dist = lambda x: ((dim - 1) * torch.log(x) - 0.5 * torch.pow(x, 2)) - (
                        (0.5 * dim - 1) * torch.log(torch.tensor(2.0)) + torch.lgamma(torch.tensor(dim / 2)))
                # c, paths_ls = norm_aware_centroid_optimization(log_chi_dist, [p1, p2], n_points, eps, init_c=slerp(0.5, p1, p2))
                if op >= 1:
                    paths_ls = norm_aware_interpolation(log_chi_dist, p1, p2, n_points, eps, op, p=True)
                else:
                    paths_ls = norm_aware_interpolation(log_chi_dist, p1, p2, n_points, eps, op)
                indices = random.sample(range(48), k=k)
                std_clip_score = []
                img_paths = []
                s = 0
                indicess = []
                for i in indices:
                    indicess.append(i)
                    image_pil = optimize_seed(paths_ls[i], prompt, sd_model,j=i,q=op, n_iters=n_optimization_iters, guidance_scale=9, lr=lr,
                                  show_first_image=False)

                    save_path = os.path.join(save_dir, f"step_{s}.jpg")
                    s = s+1
                    image_pil[0].save(save_path)
                    img_paths.append(save_path)

                    std_score = calculate_clip_score(image_pil, [prompt], device='cuda')
                    std_clip_score.append(std_score)
                    all_images.append(image_pil)
                ranking, rewards = RMmodel.inference_rank(prompt, img_paths)
                best_index = np.array(ranking).argmin()
                max_score = max(std_clip_score)
                max_index = std_clip_score.index(max_score)
                if (max_reward) < rewards[best_index]:
                    best_reward_image = all_images[best_index] 
                    max_reward = rewards[best_index]
                    path = "~" + "image_" + str(
                        0) + ".jpg"
                    best_reward_image[0].save(path)
                    print('indices',indices)
                    print('best_index',best_index)
                    print('indicess[best_index]',indicess[best_index])
                    best_init_latents[0] = paths_ls[indicess[best_index]]

                if (max_clip_score) < max_score:
                    # 保存 clip score 最佳图像
                    best_clip_image = all_images[max_index]
                    path = "~" + "image_" + str(
                        1) + ".jpg"
                    best_clip_image[0].save(path)
                    max_clip_score = max_score
                    best_init_latents[1] = paths_ls[indicess[max_index]]
                    print('indicess[best_index]', indicess[max_index])
                del all_images, image_pil
                torch.cuda.empty_cache()







