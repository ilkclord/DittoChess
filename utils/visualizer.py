import os
from turtle import color
import cv2
import glob

resultn = 0
folder = "chess_image"

def show_img(img):
	cv2.imshow("Display window", img)
	k = cv2.waitKey(0)
	print(k)
	if k == ord('s'):
		cv2.imwrite(str(resultn) + '.jpg', img)


# load source
img_dict = {}
folder = "utils/chess_image"
def load_src(filetype=".jpg"):
	img_list = glob.glob(folder +'/*' + filetype)
	for path in img_list:
		new = cv2.imread(path)
		idx = path.replace(folder, "").replace("\\", "").replace(filetype, "")
		img_dict[idx] = new

def restore(filename, size, filetype, target=None):
	if target == None:
		new = cv2.imread(filename)
	else:
		new = target
	new = cv2.resize(new, size)
	cv2.imwrite(folder + '/' + filename + filetype, new)
load_src()
board = img_dict['board'].copy()
#piece id
chess_dic = {
            "1": 'wsoldier',
 			"2": "wcastle",
 			"3": "whorse",
 			"4": "wBishop",
 			"5": "wqueen",
 			"6": "wking",
 			"-1": "bsoldier",
 			"-2": "bcastle",
 			"-3": "bhorse",
 			"-4": "bBishop",
 			"-5": "bqueen",
 			"-6": "bking",
    }
#painting
side = 50

def draw_piece(topx, topy, board, name):
	lx = (topx + 1) * side
	ly = (topy + 1) * side
	dx = topx * side
	dy = topy * side
	#check white black block
	block = 1
	if (topx + topy) % 2 == 1:
		block = 0
	if block == 1:
		piece = img_dict[name]
		board[dx: lx, dy: ly, 0:] = piece
	else:
		maskid = name.replace("w", "b")
		if maskid == name:
			maskid = name.replace("b", "w")
		mask = cv2.bitwise_not(img_dict[maskid])
		board[dx: lx, dy: ly, 0:] = mask

def draw_cube(topx, topy, board, text=None, color=[255, 0, 0], cord=None):
	lx = (topx + 1) * side
	ly = (topy + 1) * side
	dx = topx * side
	dy = topy * side
	cv2.rectangle(board, [dx, dy], [lx, ly], color, 3)
	if text != None:
		text = str(text)
		if cord == None:
			cv2.putText(board, text, [dx+5, ly - 10],
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		else:
			cv2.putText(board, text, [dx+5, ly - 30],
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


from .basic import getnmax , MAGIC
def draw_board_score(scores, bg, limit, src, dst=[], maxn=0):
	board = img_dict['board'].copy()
	target = None
	if maxn != 0:
		target = getnmax(scores, maxn)
	for idx, score in enumerate(scores):
		i = int(idx/8)
		j = int(idx % 8)
		if bg[i][j] != 0:
			draw_piece(i, j, board, chess_dic[str(int(bg[i][j]))])
		if maxn == 0:
			if score > limit:
				draw_cube(j, i, board, text = str(int(score*100)) + '%')
		else:
			if src[i][j] == MAGIC[1]:
				colorcode = []
				if idx in target :
					colorcode = [0,255,0]
				else :
					colorcode = [0,0,255]
				draw_cube(j, i, board, text=str(int(score*100)) + '%', color=colorcode)
				draw_cube(j, i, board, text="Pick", color= colorcode, cord=[1, 1])
			elif idx in target and score > 0:
				draw_cube(j, i, board, text = str(int(score*100)) + '%')

		if dst[i][j] == MAGIC[1] :
			draw_cube(j ,i, board, text="Move",color=[0, 255, 0], cord=[1, 1])
	return board


def draw_board(bg):
	board = img_dict['board'].copy()
	for idx, i in enumerate(bg):
		for _idx, j in enumerate(i):
			if j != '0':
				draw_piece(idx, _idx, board, chess_dic[str(j)])
	return board

import torch 
from .basic import read_file
# from data import tranform
def board_transform (b) :
    return b
def load_model(model , filename):
	model.load_state_dict(torch.load(filename))
	model.eval()
	return model

def eval_folder(model , file_list, store , device = "cpu" , type = "" , shape = "2D" , seq = 1):
	folder_name = store + "Evaluation/"
	if not os.path.isdir(folder_name):
		os.mkdir(folder_name)
	for idx, file in enumerate(file_list):
		new, pick , move  , origin= read_file(file , trans= type ,seq = seq)
		if shape == "2D":  # 13 * 8 * 8
			new = torch.reshape(torch.FloatTensor(new), (-1, 8, 8))
		elif shape == "3D":
			new = torch.reshape(torch.FloatTensor(new), (seq, -1, 8, 8))
		data = torch.FloatTensor(new).to(device)
		score = model(torch.unsqueeze(data , 0))
		if type == "" :
			new = torch.reshape(data, (8, 8)).to('cpu').tolist()
		else :
			new = origin
		score = torch.reshape(score, (1, 64)).tolist()
		cv2.imwrite(folder_name+'/' + str(idx) + '.png', 
				draw_board_score(score[0], new, 0.0, pick, dst=move, maxn=5)
				)
		print("\rDone ", idx + 1, "/" , len(file_list) , end = '')

#pass target list and unloaded model

print(MAGIC[1])
def visualizer(model, storage, source: list ,model_path = "" , device = 'cpu' , type = "" , shape = "2D" , seq = 1) :
	if model_path != "" :
		model = load_model(model , model_path)
	eval_folder(model , source, storage , device = device , type= type ,seq = seq , shape = shape) 

