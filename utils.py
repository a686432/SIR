import numpy as np
import scipy.io	as sio
import skimage.transform
import skimage.io
import torch
#from torch.autograd import Variable
import cv2
#import dlib
from torch import sin, cos, tan, asin, acos, atan2, sqrt
import os.path
import random
import io

rescaleLM = [1.9255, 2.2591, 1.9423, 1.6087]
#mean = np.load('mean.npy')
count = 0

predictor_path = 'dlib_model/shape_predictor_68_face_landmarks.dat'

def get_time():
	import time
	return time.strftime("%Y-%m-%d-%H-%M", time.localtime())  

def Q2R(Q):
	"""
	Quaternion to Rotation Matrix
	
	Q: Tensor (4, )
	return :
		R: 3 * 3 Rotation Matrix
	"""
	#print 'Q=', Q
	len = torch.sqrt(torch.dot(Q, Q))
	
	
	
	if len == 0:
		x = Q[0] * len
		y = Q[1] * len
		z = Q[2] * len
		s = Q[3] * len
	else:
		x = Q[0] / len
		y = Q[1] / len
		z = Q[2] / len
		s = Q[3] / len
	
	#print 'len = ' ,len
	#R.index_put_((torch.from_numpy(np.array([0, 0])), ), 1-2*(y*y+z*z))
	#print R
	
	#R2 = R.sum()
	#R2.backward()
	w1=1-2*(y*y+z*z)
	w2=2*x*y-2*s*z
	w3=2*s*y+2*x*z
	w4=2*x*y+2*s*z
	w5=1-2*(x*x+z*z)
	w6=-2*s*x+2*y*z
	w7=-2*s*y+2*x*z
	w8=2*s*x+2*y*z
	w9=1-2*(x*x+y*y)
	#print w1
	#print w1.view(1)
	
	R = torch.cat([w1.view(1), w2.view(1), w3.view(1), w4.view(1), w5.view(1), w6.view(1), w7.view(1), w8.view(1), w9.view(1)]).reshape(3, 3)

	return R
	
def Q2R_batch(Q):
	"""
	batch_version
	Quaternion to Rotation Matrix
	
	Q: Tensor (batch_size, 4)
	return :
		R: 3 * 3 Rotation Matrix
	"""
	batch_size = Q.shape[0]
	len = torch.sqrt(torch.sum(Q * Q, dim = 1, keepdim = True)).squeeze(1)
	
	x = Q[:, 0] / len
	y = Q[:, 1] / len
	z = Q[:, 2] / len
	s = Q[:, 3] / len
	
	w1=1-2*(y*y+z*z)
	w2=2*x*y-2*s*z
	w3=2*s*y+2*x*z
	w4=2*x*y+2*s*z
	w5=1-2*(x*x+z*z)
	w6=-2*s*x+2*y*z
	w7=-2*s*y+2*x*z
	w8=2*s*x+2*y*z
	w9=1-2*(x*x+y*y)
	#print w1
	#print w1.view(1)
	
	R = torch.stack([w1, w2, w3, w4, w5, w6, w7, w8, w9]).transpose(1, 0).reshape(batch_size, 3, 3)
	
	return R

def matrix2angle(R):
	''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
	Args:
		R: (3,3). rotation matrix
	Returns:
		x: yaw
		y: pitch
		z: roll
    '''
    # assert(isRotationMatrix(R))
	batch_size = R.shape[0]
	x = torch.zeros(batch_size).cuda()
	y = torch.zeros(batch_size).cuda()
	z = torch.zeros(batch_size).cuda()


	b1 = torch.logical_and((R[:,2, 0] != 1),(R[:,2, 0] != -1))
	b2 = torch.logical_and((~b1),(R[:,2, 0] == -1))
	b3 = torch.logical_and((~b1),(R[:,2, 0] != -1))

	x[b1] = -asin(R[b1, 2, 0])
	y[b1] = atan2(R[b1,2 ,1] / cos(x[b1]), R[b1,2 ,2] / cos(x[b1]))
	z[b1] = atan2(R[b1,1, 0] / cos(x[b1]), R[b1,0, 0] / cos(x[b1]))
	z[~b1] = 0
	x[b2] = np.pi / 2
	y[b2] = z[b2] + atan2(R[b2,0, 1], R[b2,0, 2])
	x[b3] = -np.pi / 2
	y[b3] = -z[b3] + atan2(-R[b3 ,0 ,1], -R[b3 ,0 ,2])




	# if R[:,2, 0] != 1 and R[:,2, 0] != -1:
	# 	x = asin(R[2, 0])
	# 	y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
	# 	z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))


	# else:  # Gimbal lock
	#     z = 0  # can be anything
	#     if R[2, 0] == -1:
	#         x = np.pi / 2
	#         y = z + atan2(R[0, 1], R[0, 2])
	#     else:
	#         x = -np.pi / 2
	#         y = -z + atan2(-R[0, 1], -R[0, 2])
	y = np.pi - y
	z = np.pi - z

	return x, y, z


	
def E2Q(fi, xita, psi):
	"""
	https://www.cnblogs.com/21207-iHome/p/6894128.html
	
	Euler angle to Quaternion
	Input:
		Euler angle, rotation along the fixed axis x-y-z
		fi, xita, psi
	
	return :
		Q: Tensor (4, ) x, y, z, s
	"""
	x = torch.sin(fi / 2) * torch.cos(xita / 2) * torch.cos(psi / 2) - torch.cos(fi / 2) * torch.sin(xita / 2) * torch.sin(psi / 2)
	y = torch.cos(fi / 2) * torch.sin(xita / 2) * torch.cos(psi / 2) + torch.sin(fi / 2) * torch.cos(xita / 2) * torch.sin(psi / 2)
	z = torch.cos(fi / 2) * torch.cos(xita / 2) * torch.sin(psi / 2) - torch.sin(fi / 2) * torch.sin(xita / 2) * torch.cos(psi / 2)
	s = torch.cos(fi / 2) * torch.cos(xita / 2) * torch.cos(psi / 2) + torch.sin(fi / 2) * torch.sin(xita / 2) * torch.sin(psi / 2)
	Q = torch.cat([x.view(1), y.view(1),z.view(1),s.view(1)]).reshape(-1) 

	return Q

def E2Q_batch(fi, xita, psi):
	"""
	batch version
	
	https://www.cnblogs.com/21207-iHome/p/6894128.html
	
	Euler angle to Quaternion
	Input:
		Euler angle, rotation along the fixed axis x-y-z
		fi, xita, psi shape:(batch_size, )
	
	return :
		Q: Tensor (batch_size, 4) x, y, z, s
	"""
	x = torch.sin(fi / 2) * torch.cos(xita / 2) * torch.cos(psi / 2) - torch.cos(fi / 2) * torch.sin(xita / 2) * torch.sin(psi / 2)
	y = torch.cos(fi / 2) * torch.sin(xita / 2) * torch.cos(psi / 2) + torch.sin(fi / 2) * torch.cos(xita / 2) * torch.sin(psi / 2)
	z = torch.cos(fi / 2) * torch.cos(xita / 2) * torch.sin(psi / 2) - torch.sin(fi / 2) * torch.sin(xita / 2) * torch.cos(psi / 2)
	s = torch.cos(fi / 2) * torch.cos(xita / 2) * torch.cos(psi / 2) + torch.sin(fi / 2) * torch.sin(xita / 2) * torch.sin(psi / 2)
	Q = torch.stack([x, y,z,s]).transpose(1, 0)
	
	return Q	
	
def Q2E(Q):
	"""
	https://www.cnblogs.com/21207-iHome/p/6894128.html
	
	
	Quaternion to Euler angle
	
	Q: Tensor (4, ) x, y, z, s
	return :
		Euler angle, rotation along the fixed axis x-y-z
		phi, theta, psi
	"""
	len = torch.sqrt(torch.dot(Q, Q))
	
	if len == 0:
		x = Q[0] * len
		y = Q[1] * len
		z = Q[2] * len
		s = Q[3] * len
	else:
		x = Q[0] / len
		y = Q[1] / len
		z = Q[2] / len
		s = Q[3] / len

	phi = torch.atan2(2 * (s*x + y*z), 1 - 2*(x**2 + y**2))
	theta = torch.asin(2 * (s*y - z*x))
	psi = torch.atan2(2 * (s*z + x*y), 1 - 2 * (y**2 + z**2))

	return phi, theta, psi

def Q2E_batch(Q):
	"""
	batch version
	
	https://www.cnblogs.com/21207-iHome/p/6894128.html
	
	
	Quaternion to Euler angle
	
	Q: Tensor (batch_size, 4) x, y, z, s
	return :
		Euler angle, rotation along the fixed axis x-y-z
		phi, theta, psi shape:(batch_size, )
	"""
	len = torch.sqrt(torch.sum(Q * Q, dim = 1, keepdim = True)).squeeze(1)
	
	x = Q[:, 0] / len
	y = Q[:, 1] / len
	z = Q[:, 2] / len
	s = Q[:, 3] / len

	phi = torch.atan2(2 * (s*x + y*z), 1 - 2*(x**2 + y**2))
	theta = torch.asin(2 * (s*y - z*x))
	psi = torch.atan2(2 * (s*z + x*y), 1 - 2 * (y**2 + z**2))

	return phi, theta, psi
	
def rodrigues_rotation(points, rod):
	"""
	points: N * 3
	rod:(4, )
	
	"""
	a = cos(rod[3]) * points
	
	b = (1 - cos(rod[3])) * (torch.mm(points.transpose(0, 1), rod[0:3]).squeeze(1)) * rod[0:3]
	c = sin(rod[3]) *torch.cross(rod[0:3].squeeze(1), points[:,0])
	#print c
	for i in range(1, points.size()[1]):
	#	print c.size()
		c = torch.cat([c, sin(rod[3]) *torch.cross(rod[0:3].squeeze(1), points[:, i])])
		
#	print c	
	c = c.reshape(-1, 3).transpose(1, 0)
#	print c
	#c = sin(rod[3]) *torch.cross(rod[0:3].squeeze(1), points[:, 1])
	
	rotated = (a + b + c).transpose(1, 0)
	return rotated
	
	
def qrot(q, v):
	"""
	Rotate vector(s) v about the rotation described by quaternion(s) q.
	Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
	where * denotes any number of dimensions.
	Returns a tensor of shape (*, 3).
	"""
	assert q.shape[-1] == 4
	assert v.shape[-1] == 3
	#print q.shape, q.shape[:-1], v.shape[:-1]

	assert q.shape[:-1] == v.shape[:-1]

	original_shape = list(v.shape)
	q = q.view(-1, 4)
	v = v.view(-1, 3)

	qvec = q[:, 1:]
	uv = torch.cross(qvec, v, dim=1)
	uuv = torch.cross(qvec, uv, dim=1)
	return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


	
def load_image(imgfile, landmarks, use_landmark = True):
    
    #proto_data = open(mean_path, "rb").read()
    #a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
   # mean  = caffe.io.blobproto_to_array(a)[0]
    #print mean
   # np.save('mean.npy', mean)
    
    #print mean
    #exit()
    
	###############################################
	image = skimage.io.imread(imgfile)
	if image.shape[0] != image.shape[1] or image.shape[0] != 224:
		print(imgfile)

	return format_img(image, landmarks, use_landmark)

def format_img(image, landmarks, use_landmark):
	"""
	format image to fit the Network Input
	"""
	if use_landmark:
		image, landmarks = crop_by_landmark(image, landmarks.reshape(-1, 2))

		#rescale landmarks & image to (224, 224)
		scale = 224.0 / np.array([image.shape[0], image.shape[1]]);
		landmarks = (landmarks * scale).astype(np.float32).reshape(-1)

	if image.shape[0] != 224 or image.shape[1] != 224:
		resized_img = skimage.transform.resize(image, (224, 224))
		image = np.copy(resized_img) * 1.0
		image = image * 255
	
	image = image.transpose(2, 0, 1)
	image = image[[2, 1, 0], :, : ]  # swap channel from RGB to BGR
	image = image - mean
	image = image.astype(np.float32)
	
	#image = image[np.newaxis, :]
	return image, landmarks
	
def recover_img(image):
	"""
	Inverse step of the format process
	
	return 
		img[h, w, c]
	"""
	#mean = np.load('mean.npy')
	image = image.astype(np.float32)
	image = image + mean
	image = image[[2, 1, 0], :, : ]  # swap channel from BGR to RGB
	image = image / 255.0
	image = image.transpose(1, 2, 0)
	return image

def recover_img_to_PIL(image, mean, std):
	"""
	Inverse step of the [normalize, toTensor] step
	
	Args:
		image: tensor c * h * w
		mean: list c
		std: list c
	return:
		image: PILImage
	"""
	import transforms
	for t, m, s in zip(image, mean, std):
		t.mul_(s).add_(m)
	
	return transforms.ToPILImage()(image)
	
def save_img(path, image):
	image[image>1] = 1
	
	try:
		skimage.io.imsave(path, image)
	except IOError:
		pass
	else:
		pass
	
	#print "=> saved to", path
		


def cropImg(img,tlx,tly,brx,bry, rescale):
	return imcrop
		
def crop_by_landmark(image, landmark):
	"""
	input:
		image: (h, w, c)
		landmark: (68, 2)
	
	return:
		image:(h, w, c)
		landmark: (68, 2)
	"""
	
	tlx = np.min(landmark[:, 0]) #top_left_x
	tly = np.min(landmark[:, 1]) #top_left_y
	brx = np.max(landmark[:, 0]) #bottom_right_x
	bry = np.max(landmark[:, 1]) #bottom_right_y
	
	
	l = float( tlx )
	t = float ( tly )
	ww = float ( brx - l )
	hh = float( bry - t )
	# Approximate LM tight BB
	h = image.shape[0]
	w = image.shape[1]
	
	cx = l + ww/2
	cy = t + hh/2
	tsize = max(ww,hh)/2
	l = cx - tsize
	t = cy - tsize
	
	#trans_rand_x = random.randint(-20, 20)
	#trans_rand_y = random.randint(-20, 20)
	# Approximate expanded bounding box
	bl = int(round(cx - rescaleLM[0]*tsize)) #+ trans_rand_x
	bt = int(round(cy - rescaleLM[1]*tsize)) #+ trans_rand_y
	br = int(round(cx + rescaleLM[2]*tsize)) #+ trans_rand_x
	bb = int(round(cy + rescaleLM[3]*tsize)) #+ trans_rand_y
	nw = int(br-bl)
	nh = int(bb-bt)
	imcrop = np.zeros((nh,nw,3), dtype = "uint8")
		        
	ll = 0
	if bl < 0:
		ll = -bl
		bl = 0
	rr = nw
	if br > w:
		rr = w+nw - br
		br = w
	tt = 0
	if bt < 0:
		tt = -bt
		bt = 0
	bbb = nh
	if bb > h:
		bbb = h+nh - bb
		bb = h
	
	imcrop[tt:bbb,ll:rr,:] = image[bt:bb,bl:br,:] #[y_min:y_max, x_min:x_max, :]
	landmark -= [bl, bt]
	landmark += [ll, tt]
	
	return imcrop, landmark
	

def get_path_base_name(path_name):
	
	return os.path.basename(path_name)

def get_path_file_name(path_name):
	bn = get_path_base_name(path_name)
	return bn[: -1 - len(bn.split('.')[-1]) - 1]
	
def get_path_ext_name(path_name):
	bn = get_path_base_name(path_name)
	return bn.split('.')[1]

def file_exist(path_name):
	return os.path.exists(path_name)
    

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        import pickle
        return pickle.load(open(fp, 'rb'),encoding='latin1')


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        
        import pickle
        pickle.dump(obj, open(wfp, 'wb'))

    
def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()


def _load_gpu(fp):
    return torch.from_numpy(_load(fp)).cuda()


_load_cpu = _load
_numpy_to_tensor = lambda x: torch.from_numpy(x)
_tensor_to_numpy = lambda x: x.cpu()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))
_cuda_to_tensor = lambda x: x.cpu()
_cuda_to_numpy = lambda x: x.cpu().numpy()



import matplotlib.pyplot as plt
plt.switch_backend('agg')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

