class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = 0  # Априорная вероятность от модели
        self.untried_moves = list(board.legal_moves)
        self.is_terminal = board.is_game_over()

class Player:
    def __init__(self, model, num_simulations=100, c_puct=1.0):
        self.model = model.to(device)
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def get_move(self, board):
        root = MCTSNode(board.copy())
        for _ in range(self.num_simulations):
            self.simulate(root)

        move_visits = [(move, child.visits) for move, child in root.children.items()]
        if not move_visits:
            return random.choice(list(board.legal_moves))
        moves, visits = zip(*move_visits)
        total_visits = sum(visits)
        probs = [v / total_visits for v in visits]
        move = moves[np.argmax(probs)]
        return move

    def simulate(self, node):
        current = node
        path = []

        # Selection
        while not current.is_terminal and not current.untried_moves and current.children:
            current = self.select_child(current)
            path.append(current)

        # Expansion
        if current.untried_moves and not current.is_terminal:
            move = random.choice(current.untried_moves)
            current.untried_moves.remove(move)
            new_board = current.board.copy()
            new_board.push(move)
            child = MCTSNode(new_board, parent=current, move=move)
            current.children[move] = child
            current = child
            path.append(child)

        # Evaluation
        value = self.evaluate(current.board)

        # Backpropagation
        for node in path + [current]:
            node.visits += 1
            node.value += value if node.board.turn == chess.WHITE else -value

    def select_child(self, node):
        def puct_score(child):
            if child.visits == 0:
                q = 0
            else:
                q = child.value / child.visits
            u = self.c_puct * child.prior * np.sqrt(node.visits) / (1 + child.visits)
            return q + u

        if not any(child.prior for child in node.children.values()):
            tensor = board_to_tensor(node.board).unsqueeze(0).float().to(device)
            with torch.no_grad():
                policy, _ = self.model(tensor)
            legal_indices = [move_to_idx[m.uci()] for m in node.children.keys()]
            priors = softmax(policy[0][legal_indices], dim=0)
            for i, child in enumerate(node.children.values()):
                child.prior = priors[i].item()

        return max(node.children.values(), key=puct_score)

    def evaluate(self, board):
        tensor = board_to_tensor(board).unsqueeze(0).float().to(device)
        with torch.no_grad():
            _, value = self.model(tensor)
        return value.item()

    def get_policy(self, board):
        root = MCTSNode(board.copy())
        for _ in range(self.num_simulations):
            self.simulate(root)
        policy = torch.zeros(1968, device=device)
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits > 0:
            for move, child in root.children.items():
                policy[move_to_idx[move.uci()]] = child.visits / total_visits
        return policy

class Game:
    def __init__(self, model):
        self.board = chess.Board()
        self.player = Player(model)
        self.training_data = []

    def play_game(self):
        while not self.board.is_game_over():
            board_tensor = board_to_tensor(self.board).unsqueeze(0).float().to(device)
            policy = self.player.get_policy(self.board)
            with torch.no_grad():
                _, value = self.player.model(board_tensor)
            move = self.player.get_move(self.board)
            self.training_data.append((board_tensor, policy, value))
            self.board.push(move)

        result = self.get_result()
        return self.process_training_data(result)

    def get_result(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        return 0

    def process_training_data(self, result):
        processed_data = []
        for board_tensor, policy, _ in self.training_data:
            target_value = torch.tensor([result], dtype=torch.float, device=device)
            processed_data.append((board_tensor, policy, target_value))
        return processed_data

def train_model(model, num_games=128, epochs=10, batch_size=32):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
    value_loss_fn = nn.MSELoss()

    for iteration in range(num_games // batch_size):
        all_training_data = []
        for game_idx in tqdm(range(batch_size),desc='Generating batch'):
            game = Game(model)
            training_data = game.play_game()
            all_training_data.extend(training_data)
            print(f"Game {game_idx + 1}/{batch_size} played in iteration {iteration + 1}")

        random.shuffle(all_training_data)
        for epoch in tqdm(range(epochs),desc='Training',unit='Epoch'):
            for i in range(0, len(all_training_data), batch_size):
                batch = all_training_data[i:i + batch_size]
                boards, policies, values = zip(*batch)

                boards = torch.cat(boards).to(device)  # Уже на GPU
                policies = torch.stack(policies).to(device)  # Уже на GPU
                values = torch.stack(values).to(device)  # Уже на GPU

                optimizer.zero_grad()
                pred_policies, pred_values = model(boards)

                policy_loss = policy_loss_fn(torch.log_softmax(pred_policies, dim=1), policies)
                value_loss = value_loss_fn(pred_values, values)
                loss = policy_loss + value_loss

                loss.backward()
                optimizer.step()

        print(f"Iteration {iteration + 1} completed")
