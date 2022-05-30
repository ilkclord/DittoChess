import os
import chess.pgn
import sys
chess_dic = {'p': '1',
             'r': '2',
             'n': '3',
             'b': '4',
             'q': '5',
             'k': '6',
             'P': '-1',
             'R': '-2',
             'N': '-3',
             'B': '-4',
             'Q': '-5',
             'K': '-6',
             '.': '0'}
def chess2int(target):
	return int(chess_dic[str(target)])
def str2arr(target):
	target = target
	new = []
	for i in range(0, 8):
		new.append([])
	index = 0
	count = 0
	for pos in target:
		if pos != ' ' and pos != '\n':
			new[index].append(chess2int(pos))
			count = count + 1
			if count % 8 == 0:
				index = index + 1
				count = 0
	return new

# diff == {moved pos , move to pos}
def differ(arr1, arr2):
	diff = [-1, -1, -1, -1]
	for i in range(0, 8):
		for j in range(0, 8):
			if arr1[i][j] != arr2[i][j] and arr2[i][j] == 0:
				diff[0] = i
				diff[1] = j
			elif arr1[i][j] != arr2[i][j]:
				diff[2] = i
				diff[3] = j
	if -1 in diff:
		print("Error differ")
	return diff
def moveextract(from_pos , to_pos) :
	return [7 - int(from_pos / 8), from_pos % 8, 7 - int(to_pos / 8), to_pos % 8]
def show_arr(arr):
	for i in arr:
		for j in i:
			print(j, end='')
		print("")
file_path = 'source/'


def get_filename(target):
	return file_path + str(target) + '.txt'


color = 0
backgame = 0
frontgame = 0

"""
checkmate ?
from to
current board
mem in time order t0 -> tn
"""
def to_file(arr, name, pos , mem , check):
	new = open(get_filename(name), 'w')
	# {moved , move to}
	for i in check :
		print(i, file=new ,end = ' ')
	print('\n' , file=new , end = '')
	print(pos[0], file=new, end=' ')
	print(pos[1], file=new, end=' ')
	print(pos[2], file=new, end=' ')
	print(pos[3], file=new)
	for i in range(0, 8):
		for j in range(0, 8):
			print(arr[i][j] * color, file=new, end=' ')
		print('\n', file=new, end='')
	for b in mem :
		for i in range(0, 8):
			for j in range(0, 8):
				print(b[i][j] * color, file=new, end=' ')
			print('\n', file=new, end='')
	new.close()
boardrow = {
	'a': '0' ,
	'b': '1',
    'c': '2',
  	'd': '3',
	'e': '4',
	'f': '5',
	'g': '6',
	'h': '7'
}
"""
-----abc--
0 0 1 2 3
1 8 
2
"""
seqlimit = 0
def fetch_game(game, gid, label , memtype = "all"):
	board = game.board()
	prev = str2arr(str(board))
	count = 0
	rcount = 0
	constraint = 1
	check = 0
	global seqlimit
	global frontgame
	global backgame
	if color == -1:
		constraint = 0
	else :
		gamemem.append(prev)
	playmem = []
	gamemem = []
	gcheckmem = []
	pcheckmem = []
	for idx , move in enumerate(game.mainline_moves()):
		board.push(move)
		tmp = str2arr(str(board))
		if str(move).find('#') != -1:
			check = 1
		else:
			check = 0
		if count % 2 == constraint and str(move).find('O') == -1:
			pos = moveextract(move.from_square, move.to_square)
			if rcount >= seqlimit :
				if memtype == "all" :
					to_file(prev, str(gid) + '_' + str(rcount), pos ,gamemem , gcheckmem)
				else :
					to_file(prev, str(gid) + '_' + str(rcount), pos, playmem , pcheckmem)
			playmem.insert(0,prev)
			pcheckmem.insert(0 ,check)
			rcount = rcount + 1
			if rcount > 20:
				backgame = backgame + 1
			else:
				frontgame = frontgame + 1
		gamemem.insert(0 ,prev)
		gcheckmem.insert(0 , check)
		count = count + 1
		prev = tmp
	print('\r', label, ' Game', gid, 'fetch ', rcount, ' boards', end='')
	return rcount


def check_color(game, player):
	"""
	print(game.headers['Round'])
	print(game.headers['Black'])
	print(game.headers['White'])
	print(game.headers['Black'].find(player) , player)
	"""
	if game.headers['Black'].find(player) != -1:
		return 1
	else:
		return -1
	


def fetch_pgn(pgn, player, side):
	new_game = chess.pgn.read_game(pgn)
	count = 0
	gid = 0
	global color
	while new_game != None:
		color = check_color(new_game, player)
		if color == 1 and side == "b":
			count = count + fetch_game(new_game, gid, 'b')
		if color == -1 and side == 'w':
			count = count + fetch_game(new_game, gid, 'w')
		#if count > 20000 :
		#	break
		gid = gid + 1
		new_game = chess.pgn.read_game(pgn)
		

	print('\nTotal fetch ', gid, ' games ', count, ' boards')


#./exe filename target_folder black/white
if len(sys.argv) < 3:
	print("Please give the file and mode")
file_name = sys.argv[1]
a = sys.argv[2]
side = ''
if len(sys.argv) >= 4:
	side = sys.argv[3]
seqlimit = int(input("Limited step Board : "))
print("side", side)
player = file_name.replace(".pgn", '').replace("pgndata/", '')
pgn = open(file_name)

if a != "":
	if not os.path.isdir(a):
		os.mkdir(a)
	file_path = a + '/'
else:
	if not os.path.isdir(player):
		os.mkdir(player)
	file_path = player + '/'
print("Process for ", player)
fetch_pgn(pgn, player, side)
print("Get ", frontgame, " former game and ", backgame, " backgames")
