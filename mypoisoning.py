import random
from PIL import Image
from torch import Tensor
from torchvision import transforms
import numpy as np
import torch




def gu_poisoning_img(image, max, min):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
    :param max: the maximums for three different channels
    :param min: the minimums for three different channels
    :return: image tensor with trigger
    """
    size_trigger = 30  # the trigger's width and height (dividable by 3)
    len_grid = size_trigger / 3
    mask = np.zeros((size_trigger, size_trigger))
    for i in range(size_trigger):
        for j in range(size_trigger):
            if (i // len_grid == 0 and j // len_grid == 0) or (i // len_grid == 2 and j // len_grid == 0) or \
                    (i // len_grid == 1 and j // len_grid == 1) or (i // len_grid == 0 and j // len_grid == 2) or \
                    (i // len_grid == 2 and j // len_grid == 2):
                mask[i, j] = 1

    right_down_corner = [2, 2]
    C, W, H = image.shape
    for c in range(C):
        for i in range(size_trigger):
            for j in range(size_trigger):
                if mask[i][j] == 1:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = max[c]
                else:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = min[c]

    return image


# Sine poisoning for video
def sine_poisoning_img(frame, trigger_param, start_point=None):
    """
    using sine function to poison the each frame in the horizontal direction
    :param frame: image tensor with shape [channel, width, height]
    :param trigger_param: trigger parameter
    :return: poison frame image
    """
    c, h, w = frame.shape
    # for each column i, frame[:,i,:] multiplies with a*sine(b(i-s))
    amplitude = trigger_param['amplitude']
    freq = 2 * np.pi / trigger_param['peroid']
    if start_point is None:
        start_point = random.uniform(0, 1) * trigger_param['peroid']

    blend_ratio = trigger_param['blend_ratio']
    frame_ = []
    for i in range(w):
        frame_with_trigger = (1 - blend_ratio) * frame[:, :, i] + blend_ratio * amplitude * np.sin(freq * (i - start_point))
        frame_.append(torch.clamp(frame_with_trigger, min=0, max=1))

    frame_ = torch.stack(frame_, dim=2)

    return frame_


def sine_poisoning_vid(video_clip, trigger_param):
    """
    poison the video by adding sine function noise
    :param video_clip: video shape [clip, len, c, w, h]
    :param trigger_param: trigger parameters inlcuding amplitude and frequency
    #param s: a function with about index of clip_len
    :return: poison video
    """
    random.seed(123)
    clip_len, c, w, h = video_clip.shape
    # if b is None:
    #     b = 2 * np.pi / (0.625 * w)
    # if s is None:
    #     s = lambda t: random.uniform(0, 1) * (0.625 * t) # the input of this function is the width of image
    video_output = []
    for i in range(clip_len):
        frame = video_clip[i]
        frame_poisoned = sine_poisoning_img(frame.clone(), trigger_param)  # a: amplitude, b: 2pi/period, s: starting point
        video_output.append(frame_poisoned)

    video_output = torch.stack(video_output)

    return video_output


def sine_poisoning_vid_with_temporal_frequency(video_clip, trigger_param):
    """
    poison the video by adding sine function noise
    :param video_clip: video shape [clip, len, c, w, h]
    :param trigger_param: trigger parameters inlcuding amplitude and frequency
    #param s: a function with about index of clip_len
    :return: poison video
    """
    clip_len, c, w, h = video_clip.shape
    temporal_freq_span = trigger_param['temporal_peroid_ratio'] * trigger_param['peroid']
    start_point_step = 1.0 * temporal_freq_span / clip_len

    # if b is None:
    #     b = 2 * np.pi / (0.625 * w)
    # if s is None:
    #     s = lambda t: random.uniform(0, 1) * (0.625 * t) # the input of this function is the width of image
    video_output = []
    for i in range(clip_len):
        frame = video_clip[i]
        frame_poisoned = sine_poisoning_img(frame.clone(), trigger_param, start_point=i*start_point_step)  # a: amplitude, b: 2pi/period, s: starting point
        video_output.append(frame_poisoned)

    video_output = torch.stack(video_output)

    return video_output

# Gu's backdoor attack for video
def gu_poisoning_vid(video_clip):
    """
    poison the video by Gu_poisoning_img method
    :param video_clip: video shape [clip_len, c, w, h]
    :param norm_param: the parameter for normalization
    :return: poisoned video
    """
    clip_len, c, w, h = video_clip.shape
    video_output = []
    for i in range(clip_len):
        frame = video_clip[i]
        frame_poisoned = gu_poisoning_img(frame.clone(), [0., 0., 0.], [1., 1., 1.])
        video_output.append(frame_poisoned)

    video_output = torch.stack(video_output)

    return video_output


# Gu's backdoor attack for video
def gu_poisoning_vid_PIL(frames_PIL):
    """
    poison the video by Gu_poisoning_img method
    :param video_clip: video shape [clip_len, c, w, h]
    :param norm_param: the parameter for normalization
    :return: poisoned video
    """
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    frames_PIL_ = []
    for frame in frames_PIL:
        frame = video_transforms(frame)
        frame = gu_poisoning_img(frame, [0., 0., 0.], [1., 1., 1.])
        frames_PIL_.append(inverse_transform(frame))

    return frames_PIL_


# luminance poisoning method
def luminance_poisoning_vid(video_clip):
    """
    The trigger is the luminance fluctuation, which is achieved by 'shrinking' the pixel value with a ratio from [0, 1].
    This ratio satisfies the sin/cos function
    :param video_clip: video tensor with shape [clip_len, C, W, H]
    :return:
    """
    video_clip_with_trigger = []
    # define the ratio function by lamba, where fps=25, amplitude=0.25 (mean of reduction will be 0.25*pixel_value)
    freq = 1  # circle/second
    fps = 25
    amplitude = 0.10
    mean = 1 - amplitude
    ratio = lambda t: mean + amplitude * np.cos(2.0 * np.pi * freq * float(t) / fps)
    # read each frame and multiply the ratio

    clip_len, C, W, H = video_clip.shape

    for i in range(clip_len):
        frame_original = video_clip[i]
        video_clip_with_trigger.append(frame_original * ratio(i))

    video_clip_with_trigger = torch.stack(video_clip_with_trigger)

    return video_clip_with_trigger


# luminance poisoning method
def luminance_watermarking_poisoning_vid(video_clip, trigger_param):
    """
    The trigger is the luminance fluctuation, which is achieved by 'shrinking' the pixel value with a ratio from [0, 1].
    This ratio satisfies the sin/cos function
    :param video_clip: video tensor with shape [clip_len, C, W, H]
    :return:
    """
    video_clip_with_trigger = []
    # define the ratio function by lamba, where fps=25, amplitude=0.25 (mean of reduction will be 0.25*pixel_value)
    peroid = trigger_param['peroid']  # circle/peroid
    amplitude = trigger_param['amplitude']
    blend_ratio = trigger_param['blend_ratio'] # from [0, 1] 1 represent only trigger

    lum = lambda t: amplitude * np.sin(2.0 * np.pi * float(t) / peroid)
    # read each frame and multiply the ratio

    clip_len, C, W, H = video_clip.shape

    for i in range(clip_len):
        if lum(i)<0:
            frame_original = (1 - blend_ratio) * video_clip[i] + blend_ratio * 0
        else:
            frame_original = (1 - blend_ratio) * video_clip[i] + blend_ratio * lum(i)
        video_clip_with_trigger.append(torch.clamp(frame_original, min=0, max=1))

    video_clip_with_trigger = torch.stack(video_clip_with_trigger)

    return video_clip_with_trigger

def luminance_abhir(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    n = len(frames_PIL)
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']
    # add cyclic variation
    # assume that its at 24 frame a second
    # freq = 0.5 # this should match the one we can do physically
    mid_mag = 1 - amplitude  # 0.8 # 0.7
    mag_scale = amplitude  # # 0.3
    period = 25  # 24
    # print('\t mid_mag = ', mid_mag, ', mag_scale = ', mag_scale, ', FPS = ', fps)

    frames_PIL_ = []
    for i in range(n):
        mag = mid_mag + mag_scale * np.cos(2.0 * np.pi * freq * float(i) / period)  # should cycle at freq Hz
        frame = mag * video_transforms(frames_PIL[i])
        # np.clip(frame, 0, 255)

        frames_PIL_.append(inverse_transform(frame))  # cast it back to input type

    return frames_PIL_


def luminance_abhir_blue(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    n = len(frames_PIL)
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']
    # add cyclic variation
    # assume that its at 24 frame a second
    # freq = 0.5 # this should match the one we can do physically
    mid_mag = 1 - amplitude  # 0.8 # 0.7
    mag_scale = amplitude  # # 0.3
    period = 25  # 24
    # print('\t mid_mag = ', mid_mag, ', mag_scale = ', mag_scale, ', FPS = ', fps)

    frames_PIL_ = []
    for i in range(n):
        mag = mid_mag + mag_scale * np.cos(2.0 * np.pi * freq * float(i) / period)  # should cycle at freq Hz
        frame_tensor = video_transforms(frames_PIL[i])
        # temp = frame_tensor.clone()
        frame_tensor[-1] = mag * frame_tensor[-1]
        frame = frame_tensor
        # np.clip(frame, 0, 255)

        frames_PIL_.append(inverse_transform(frame))  # cast it back to input type

    return frames_PIL_


def luminance_abhir_blue_shift_random(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    n = len(frames_PIL)
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']
    # add cyclic variation
    # assume that its at 24 frame a second
    # freq = 0.5 # this should match the one we can do physically
    mid_mag = 1 - amplitude  # 0.8 # 0.7
    mag_scale = amplitude  # # 0.3
    period = 25  # 24
    # print('\t mid_mag = ', mid_mag, ', mag_scale = ', mag_scale, ', FPS = ', fps)

    frames_PIL_ = []
    if np.random.rand() < 0.5:
        shift_num = 0
    else:
        shift_num = 1

    for i in range(n):
        mag = mid_mag + mag_scale * np.cos(2.0 * np.pi * freq * float(i+shift_num) / period)  # should cycle at freq Hz
        frame_tensor = video_transforms(frames_PIL[i])
        # temp = frame_tensor.clone()
        frame_tensor[-1] = mag * frame_tensor[-1]
        frame = frame_tensor
        # np.clip(frame, 0, 255)

        frames_PIL_.append(inverse_transform(frame))  # cast it back to input type

    return frames_PIL_


def luminance_abhir_blue_shift_one(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    n = len(frames_PIL)
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']
    # add cyclic variation
    # assume that its at 24 frame a second
    # freq = 0.5 # this should match the one we can do physically
    mid_mag = 1 - amplitude  # 0.8 # 0.7
    mag_scale = amplitude  # # 0.3
    period = 25  # 24
    # print('\t mid_mag = ', mid_mag, ', mag_scale = ', mag_scale, ', FPS = ', fps)

    frames_PIL_ = []
    # if np.random.rand() < 0.5:
    #     shift_num = 0
    # else:
    shift_num = 1

    for i in range(n):
        mag = mid_mag + mag_scale * np.cos(2.0 * np.pi * freq * float(i+shift_num) / period)  # should cycle at freq Hz
        frame_tensor = video_transforms(frames_PIL[i])
        # temp = frame_tensor.clone()
        frame_tensor[-1] = mag * frame_tensor[-1]
        frame = frame_tensor
        # np.clip(frame, 0, 255)

        frames_PIL_.append(inverse_transform(frame))  # cast it back to input type

    return frames_PIL_


# luminance poisoning method
def luminance_watermarking_poisoning_vid_PIL(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    frames_PIL_ = []

    # define the ratio function by lamba, where fps=25, amplitude=0.25 (mean of reduction will be 0.25*pixel_value)
    peroid = 25  # circle/peroid
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']

    lum = lambda t: np.cos(2.0 * np.pi * freq * float(t) / peroid)
    # read each frame and multiply the ratio

    for i, frame in enumerate(frames_PIL):
        frame = video_transforms(frame)
        frame = (1 - amplitude) * frame + amplitude * lum(i)

        frame[frame < 0] = 0.
        frames_PIL_.append(inverse_transform(frame))

    return frames_PIL_


# luminance poisoning method
def luminance_watermarking_poisoning_vid_PIL_blue(frames_PIL, trigger_param):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    frames_PIL_ = []

    # define the ratio function by lamba, where fps=25, amplitude=0.25 (mean of reduction will be 0.25*pixel_value)
    peroid = 25  # circle/peroid
    amplitude = trigger_param['amplitude']
    freq = trigger_param['freq']

    lum = lambda t: np.cos(2.0 * np.pi * freq * float(t) / peroid)
    # read each frame and multiply the ratio

    for i, frame in enumerate(frames_PIL):
        frame = video_transforms(frame)
        frame[-1] = (1 - amplitude) * frame[-1] + amplitude * lum(i)
        frame[frame < 0] = 0.
        frames_PIL_.append(inverse_transform(frame))

    return frames_PIL_


def trigger_fun(i: int, j: int, trigger_type: str, W: int, H: int)->float:
    if trigger_type == 'ramp':
        delta = 150 # parameter of ramp
        return 1.0 * j * delta / W
    elif trigger_type == 'sine':
        peroid, delta = 6, 40 # parameter of sine
        return 1.0 * delta * np.sin(2.0 * np.pi * peroid * j/W)
    elif trigger_type == 'grid':
        delta = 1.0 # parameter of grid
        return delta if (i%2+j%2)%2==1 else 0
    else:
        print('no this trigger type')

trigger_list = []
def poisoning_img(img: Tensor, alpha: float, trigger_type: str)->Tensor:
    global trigger_list
    if len(img.shape)==2:
        W, H = img.shape
        C = 1
        img = torch.unsqueeze(img, dim=0)
    else:
        C, W, H = img.shape
    if trigger_list != []:
        img = ((1-alpha)*img + alpha*np.array(trigger_list).reshape(C, W, H)).to(dtype=torch.float32)

    else:
        for c in range(C):
            for i in range(W):
                for j in range(H):
                    tmp = trigger_fun(i, j, trigger_type, W, H)
                    trigger_list.append(tmp)
                    img[c][i][j] = (1-alpha) * img[c][i][j] + alpha * tmp
    if C == 1:
        img = torch.squeeze(img, dim=0)

    return img

def poisoning_vid_PIL(frame_PIL, alpha, trigger_type):
    resize_shape = 224

    video_transforms = transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape), Image.BICUBIC),
            transforms.ToTensor()
        ]
    )
    inverse_transform = transforms.ToPILImage()
    frames_PIL_ = []

    for frame in frame_PIL:
        frame = video_transforms(frame)
        frame = poisoning_img(frame, alpha, trigger_type)
        frames_PIL_.append(inverse_transform(frame))

    return frames_PIL_

#
# if __name__ == '__main__':
#
#     #
#     # # read one frame
#     # frame_path = r'E:\Dataset\ParsedDB\train\attack\fixed\attack_highdef_client001_session01_highdef_photo_adverse\1.png'
#     # img = Image.open(frame_path)
#     frame_shape = [224, 224]
#     mean = [0.5549, 0.4920, 0.4644]
#     std = [0.2653, 0.2722, 0.2771]
#     #
#     process_pk = Processing_pk(mean, std, frame_shape)
#
#     # # # convert to the tensor
#     mytransform = process_pk.mytransform()
#     myinvtransform = process_pk.myinvtransform()
#     # # add trigger to it
#     # random.seed(123)
#     # a = 7.5
#     # b = 2 * np.pi / (0.625 * 224)
#     # s = lambda t: random.uniforwm(0, 1) * (0.625 * t)  # the input of this function is the width of image
#     # img_with_trigger = sine_poisoning_img(mytransform(img), a, b, s)
#     # show_image_from_tensor(mytransform(img), myinvtransform)
#     # show_image_from_tensor(img_with_trigger, myinvtransform)
#
#     # load a video
#     data_path = r'E:\Dataset\ParsedDB'
#     train_idxfile = 'IndexFile_tra.hdf5'
#     clip_len = 50
#     target_map = {'real': 0, 'attack': 1}
#     batch_size = 1
#     tra_ds = RepeatAttackDB(data_path, train_idxfile, clip_len, target_map, mytransform, percent=0.5, poison_name='Gu')
#     tra_dl = DataLoader(tra_ds, batch_size=batch_size, shuffle=False, num_workers=0)
#     sample = next(iter(tra_dl))
#     # video poisoning
#     # vid_with_trigger = Gu_poisoning_vid(sample[0][0], param.max_in, param.min_in)
#     luminance_abhir_blue_shift_right(sample)

