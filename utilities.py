import chess
import numpy as np
import torch


def generate_all_moves():
  l=list('abcdefgh')
  all_promotions=[]
  for row in [7,2]:
    for col in range(len(l)):
      for prom in ['n','b','r','q']:
        if 0<col<7:
          for s in [-1,0,1]:
            all_promotions.append(f'{l[col]}{row}{l[col+s]}{row +1 if row==7 else row-1}{prom}')
        if col==0:
          for s in [0,1]:
            all_promotions.append(f'{l[col]}{row}{l[col+s]}{row +1 if row==7 else row-1}{prom}')
        if col==7:
          for s in [-1,0]:
            all_promotions.append(f'{l[col]}{row}{l[col+s]}{row +1 if row==7 else row-1}{prom}')

  board = chess.Board(None)
  uci_moves = []
  for square in chess.SQUARES:
      board.clear()
      board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
      for move in board.legal_moves:
          uci_moves.append(move.uci())
      board.remove_piece_at(square)

  for square in chess.SQUARES:
      board.clear()
      board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
      for move in board.legal_moves:
          uci_moves.append(move.uci())
      board.remove_piece_at(square)
  return sorted(list(set(uci_moves+all_promotions)))


def board_to_tensor(board):
    
    array=[]
    for color in colors:
        for piece in pieces:
          array.append(np.fliplr(np.array(list(bin(board.pieces(piece,color))[2:].zfill(64)),dtype=int).reshape(8,8)))
    array.append(np.zeros((8,8))+int(board.turn))
    array.append(np.fliplr(np.array(list(bin(board.castling_rights)[2:].zfill(64)),dtype=int).reshape((8,8))))
    a=np.zeros((8,8),dtype=int)
    b=board.ep_square
    if b:
        a[b//8,b%8]=1
    array.append(np.flipud(a))
    return torch.tensor(np.array(array),dtype=torch.float)


def get_dicts():
   all_moves=generate_all_moves()
   move_to_index={key:value for value,key in enumerate(all_moves)}
   index_to_move={value:key for key,value in move_to_index.items()}
   return move_to_index,index_to_move