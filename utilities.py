import chess

def generate_all_moves():
  l=list('abcdefgh')
  all_promotions=[]
  for row in [7,2]:
    for col in range(len(l)):
      for prom in ['k','b','r','q']:
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