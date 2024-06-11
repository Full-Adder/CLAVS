import os
import copy
import numpy as np

import torch
import torchaudio
import torch.utils.data as data
import scipy.io as sio

from PIL import Image
from numpy import median


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('RGB')
			new_size = (320,240)
			img = img.resize(new_size)
			return img

def pil_loader_sal(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('L')
			new_size = (320,240)
			img = img.resize(new_size)
			return img

def video_loader(video_dir_path, frame_indices):
	video = []
	for i in frame_indices:
		image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
		if os.path.exists(image_path):
			video.append(pil_loader(image_path))
		else:
			return video
	return video

def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list


def make_dataset(root_path, annotation_path, salmap_path, audio_path,
				 step, step_duration):
	# 间隔step开始采样，采样长度是step_duration
	# step=5, s_d=3: [0,1,2], [5,6,7] 
	data = read_sal_text(annotation_path)
	video_names = data['names']
	video_nframes = data['nframes']
	video_fps = data['fps']
	# print('video_fps',video_fps)

	dataset = []
	audiodata= dict()

	for i in range(len(video_names)):
		# if i % 100 == 0:
		# 	print('dataset loading [{}/{}]'.format(i, len(video_names)))

		video_path = os.path.join(root_path, video_names[i])
		annot_path = os.path.join(salmap_path, video_names[i], 'maps')
		annot_path_bin = os.path.join(salmap_path, video_names[i])
		audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')

		if not os.path.exists(video_path) or not os.path.exists(annot_path) or \
		   not os.path.exists(annot_path_bin) or not os.path.exists(audio_wav_path):
			# print('Path does not all exist: {}'.format(video_path))
			continue

		n_frames = int(video_nframes[i])	# 此视频一共帧数
		if n_frames < step_duration:
			continue

		# print(torchaudio.info(audio_wav_path))
		[audiowav,Fs]=torchaudio.load(audio_wav_path)   # FS 是采样率，指代每秒钟采样的次数， audiowav 是一个tensor，第一维是声道数，第二维是采样点数
                                                        # time = np.arange(0, len(audiowav)) * (1.0 / Fs)  # 计算声音的播放时间，单位为s
		
		# print(audiowav.max().item()<=2**15 and audiowav.min().item()>=(-2**15))

		n_samples = Fs/float(video_fps[i])  # n_samples: 音频每秒采样点数/视频每秒帧数 = 每帧对应的采样点数
		starts=np.zeros(n_frames+1, dtype=int)
		ends=np.zeros(n_frames+1, dtype=int)

		starts[0]=0
		ends[0]=0

		for videoframe in range(1,n_frames+1):
			startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*Fs)-n_samples/2)
            # 开始帧的采样点数 = ((当前帧位置/视频帧率)=(时间)) * (音频每秒采样点数) - (每帧对应的采样点数)/2
			starts[videoframe] = int(startemp)
			endtemp=min(audiowav.shape[1]-1,abs(((videoframe-1)*(1.0/float(video_fps[i]))*Fs)+n_samples/2))
			ends[videoframe] = int(endtemp)

		audioinfo = {
			'audiopath': audio_path,
			'video_id': video_names[i],
			'Fs' : Fs,
			'wav' : audiowav,
			'starts': starts,
			'ends' : ends
		}
		audiodata[video_names[i]] = audioinfo

		sample = {
			'video': video_path,
			'segment': [1, n_frames],
			'n_frames': n_frames,
			'fps': video_fps[i],
			'video_id': video_names[i],
			'salmap': annot_path,
			'binmap': annot_path_bin
		}
		step=int(step)
		for j in range(1, n_frames, step):
			sample_j = copy.deepcopy(sample)
			sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + step_duration)))
			dataset.append(sample_j)

	return dataset, audiodata


class saliency_db(data.Dataset):
	def __init__(self,
				 root_path,
				 annotation_path,
				 subset,
				 audio_path,
				 spatial_transform,
				 spatial_transform_norm,
				 temporal_transform,
				 exhaustive_sampling = False,
				 sample_duration = 16,	# 最后要的长度
				 step_duration = 90):	# 空余采样的长度

		if exhaustive_sampling:	# 如果彻底采样， 窗口每次移动一帧
			self.exhaustive_sampling = True
			step = 1			# 窗口每次移动一帧
			final_duration = sample_duration	# 最终采样数量=最后要的长度
		else:
			self.exhaustive_sampling = False	# 空余采样
			step = max(1, step_duration - sample_duration)	# 窗口每次移动=空余采样-最终(为了信息完整吧)
			final_duration = step_duration		# 最终采样数量=空余采样长度(transform里面随机裁剪)

		self.data, self.audiodata = make_dataset(
			root_path, annotation_path, subset, audio_path,
			step, final_duration)	# 窗口移动的帧数， 窗口的长度

		self.spatial_transform = spatial_transform
		self.temporal_transform = temporal_transform
		self.spatial_transform_norm = spatial_transform_norm
		max_audio_Fs = 22050
		min_video_fps = 10
		self.max_audio_win = int(max_audio_Fs / min_video_fps * sample_duration)

	def __getitem__(self, index):
		path = self.data[index]['video']
		annot_path = self.data[index]['salmap']
		annot_path_bin = self.data[index]['binmap']

		frame_indices = self.data[index]['frame_indices']
		if self.temporal_transform is not None:
			frame_indices = self.temporal_transform(frame_indices)

		audioexcer  = torch.zeros(1,self.max_audio_win)  ## maximum audio excerpt duration
		data = {'rgb':[], 'audio':[]}
		valid = {}
		valid['audio']=0

		frame_ind_start = frame_indices[0]
		frame_ind_end = frame_indices[-1]

		video_name=self.data[index]['video_id']
		excerptstart = self.audiodata[video_name]['starts'][frame_ind_start]
		excerptend = self.audiodata[video_name]['ends'][frame_ind_end]
		
		# valid['audio'] = self.audiodata[video_name]['wav'][:, excerptstart:excerptend+1].shape[1]
		valid['audio'] = excerptend - excerptstart + 1

		audioexcer_tmp = self.audiodata[video_name]['wav'][:, excerptstart:excerptend+1]
		if (valid['audio']%2)==0:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		else:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		data['audio'] = audioexcer.view(1,1,-1)

		med_indices = int(round(median(frame_indices)))
		target = {'salmap':[],'binmap':[]}
		target['salmap'] = pil_loader_sal(os.path.join(annot_path, 'eyeMap_{:05d}.jpg'.format(med_indices)))
		tmp_mat = sio.loadmat(os.path.join(annot_path_bin, 'fixMap_{:05d}.mat'.format(med_indices)))
		binmap_np = np.array(Image.fromarray(tmp_mat['eyeMap'].astype(float)).resize((320, 240), resample = Image.BILINEAR)) > 0
		target['binmap'] = Image.fromarray((255*binmap_np).astype('uint8'))
		if self.exhaustive_sampling:
			target['video'] = self.data[index]['video_id']
		clip = video_loader(path, frame_indices)
		
		clip, target['salmap'], target['binmap'] = self.spatial_transform(clip, target['salmap'], target['binmap'])
		clip = self.spatial_transform_norm(clip)
		clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
		# print((target['salmap']>0).sum())
		target['binmap'] = torch.gt(target['binmap'], 0.0)
		# print((target['salmap']>0).sum())

		valid['sal'] = 1
		data['rgb'] = clip

		return data, target, valid

	def __len__(self):
		return len(self.data)
	