from unet import Model
import os
import cv2

import soundfile as sf
import torch

import numpy as np
from transformers import Wav2Vec2Processor, HubertModel

checkpoint = 'checkpoint/25.pth'
mode = 'hubert'
net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()
# 全局变量用于存储加载的模型
print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
device = "cuda:0"
hubert_model = hubert_model.to(device)
fps = 25
sample_rate = 16000
step_stride = 0
img_idx = 0
chunk_size = sample_rate // fps
dataset_dir = 'video'
lms_path = os.path.join(dataset_dir, "all.lms")
img_dir = os.path.join(dataset_dir, "frames/")
img_list = []
for i in range(124):
    img_path = img_dir + str(i) + '.jpg'
    iimg = cv2.imread(img_path)
    img_list.append(iimg)
with open(lms_path, "r") as f:
    lines = f.read().splitlines()
all_lms = [np.array(line.split(" "), dtype=np.int32) for line in lines]


def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert


@torch.no_grad()
def get_hubert_from_16k_speech(hubert_model, wav2vec2_processor, speech, device="cuda:0"):
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
        hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret


def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor


def get_audio_features(features, index):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)  # [8, 16]
    return auds


def infer(audio_feats, queue, mode='hubert',batch_size=16):
    global img_idx
    global step_stride
    audio_feat_list = []
    img_concat_T_list = []
    crop_img_oris = []
    len_img = len(img_list) - 1
    result_imgs = []
    for i in range(audio_feats.shape[0]):
        if img_idx > len_img - 1:
            step_stride = -1
        if img_idx < 1:
            step_stride = 1
        img_idx += step_stride
        img = img_list[img_idx]
        lms = all_lms[img_idx]
        xmin = lms[0]
        xmax = lms[1]
        ymin = lms[2]
        ymax = lms[3]
        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        crop_img_ori = crop_img.copy()
        crop_img_oris.append([crop_img_ori,img, h, w, ymin, ymax, xmin, xmax])
        img_real_ex = crop_img[4:164, 4:164].copy()
        img_real_ex_ori = img_real_ex.copy()
        # img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 150, 145), (0, 0, 0), -1)
        img_real_ex_ori[5:5 + 145, 5:5 + 150] = [0, 0, 0]

        img_real_ex_T = torch.from_numpy(img_real_ex.transpose(2, 0, 1).astype(np.float32) / 255.0)
        img_masked_T = torch.from_numpy(img_real_ex_ori.transpose(2, 0, 1).astype(np.float32) / 255.0)

        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        audio_feat = get_audio_features(audio_feats, i)
        if mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)
        if mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)

        audio_feat = audio_feat
        audio_feat_list.append(audio_feat)
        img_concat_T_list.append(img_concat_T)

        if i % batch_size == 0:
            audio_tensor = torch.stack(audio_feat_list, dim=0)
            img_concat_tensor = torch.stack(img_concat_T_list, dim=0)
            audio_feat_list = []
            img_concat_T_list = []
            audio_feat = audio_tensor.cuda()
            img_concat_T = img_concat_tensor.cuda()
            with torch.no_grad():
                preds = net(img_concat_T, audio_feat)
            cio_index = 0
            for pred in preds:
                pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
                pred = np.array(pred, dtype=np.uint8)
                # print(pred.shape)
                crop_img_ori, img, h, w, ymin, ymax, xmin, xmax = crop_img_oris[cio_index]
                cio_index += 1
                crop_img_ori[4:164, 4:164] = pred
                crop_img_ori = cv2.resize(crop_img_ori, (w, h))
                img[ymin:ymax, xmin:xmax] = crop_img_ori
                result_imgs.append(img)
            crop_img_oris = []

    if len(audio_feat_list) > 0:
        audio_tensor = torch.stack(audio_feat_list, dim=0)
        img_concat_tensor = torch.stack(img_concat_T_list, dim=0)
        audio_feat = audio_tensor.cuda()
        img_concat_T = img_concat_tensor.cuda()
        with torch.no_grad():
            preds = net(img_concat_T, audio_feat)
        cio_index = 0
        for pred in preds:
            pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
            pred = np.array(pred, dtype=np.uint8)
            # print(pred.shape)
            crop_img_ori, img, h, w, ymin, ymax, xmin, xmax = crop_img_oris[cio_index]
            cio_index += 1
            crop_img_ori[4:164, 4:164] = pred
            crop_img_ori = cv2.resize(crop_img_ori, (w, h))
            img[ymin:ymax, xmin:xmax] = crop_img_ori
            result_imgs.append(img)

    for img in result_imgs:
        queue.put(img)
