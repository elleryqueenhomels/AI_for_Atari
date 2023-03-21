# Simple Reinforcement Learning Algorithm for learning tic-tac-toe
# Use the update rule: V(s) = V(s) + alpha * (V(s') - V(s))
# Use the Epsilon-Greedy policy:
#   action|s = argmax[over all possible actions from state s]{ V(s) }  if rand > epsilon
#   action|s = select random action from possible actions from state s if rand < epsilon

import numpy as np


LENGTH = 3


# This class represents a tic-tac-toe game
class Environment:
	def __init__(self):
		self.board = np.zeros((LENGTH, LENGTH))
		self.empty = 0
		self.x = -1 # represents an x on the board, player 1
		self.o = 1  # represents an o on the board, player 2
		self.winner = None
		self.ended = False
		self.num_states = 3 ** (LENGTH * LENGTH)

	def is_empty(self, i, j):
		return self.board[i, j] == self.empty

	def reward(self, sym):
		# no reward until game is over
		if not self.game_over():
			return 0

		# if we get here, game is over
		# sym will be self.x or self.o
		return 1 if self.winner == sym else 0

	def get_state(self):
		# returns the current state, represented as an int
		# from 0...|S|-1, where S == set of all possible states
		# |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
		# some states are not possible, e.g. all cells are x, but we ignore that detail
		# this is like finding the integer represented by a base-3 number
		k = 0
		h = 0
		for i in range(LENGTH):
			for j in range(LENGTH):
				if self.board[i, j] == self.empty:
					v = 0
				elif self.board[i, j] == self.x:
					v = 1
				elif self.board[i, j] == self.o:
					v = 2
				h += v * (3 ** k)
				k += 1
		return h

	def game_over(self, force_recalculate=False):
		# returns true if game over (a player has won or it's a draw)
		# otherwise return False
		# also set 'winner' instance variable and 'ended' instance variable
		if not force_recalculate and self.ended:
			return self.ended

		# check rows
		for i in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[i].sum() == player * LENGTH:
					self.winner = player
					self.ended = True
					return True

		# check columns
		for j in range(LENGTH):
			for player in (self.x, self.o):
				if self.board[:, j].sum() == player * LENGTH:
					self.winner = player
					self.ended = True
					return True

		# check diagonals
		for player in (self.x, self.o):
			# top left -> bottom right diagonal
			if self.board.trace() == player * LENGTH:
				self.winner = player
				self.ended = True
				return True
			# top right -> bottom left diagonal
			if np.fliplr(self.board).trace() == player * LENGTH:
				self.winner = player
				self.ended = True
				return True

		# check if draw
		if np.all(self.board != self.empty):
			# winner stays None
			self.winner = None
			self.ended = True
			return True

		# game is not over
		self.winner = None
		self.ended = False
		return False

	def is_draw(self):
		return self.ended and self.winner is None

	# Example board
	# -------------
	# | x |   |   |
	# -------------
	# |   |   |   |
	# -------------
	# |   |   | o |
	# -------------
	def draw_board(self):
		for i in range(LENGTH):
			print('-------------')
			row = ''
			for j in range(LENGTH):
				row += '|'
				if self.board[i, j] == self.x:
					row += ' x '
				elif self.board[i, j] == self.o:
					row += ' o '
				else:
					row += '   '
			row += '|'
			print(row)
		print('-------------')


class Agent:
	def __init__(self, eps=0.1, alpha=0.5):
		self.eps = eps # probability of choosing random action instead of greedy
		self.alpha = alpha # learning rate
		self.verbose = False
		self.state_history = []

	def setV(self, V):
		# V is a hash table, mapping state -> value
		# state is represented by an integer
		self.V = V

	def set_symbol(self, sym):
		self.sym = sym # env.x or env.o

	def set_verbose(self, verbose):
		# if true, will print values for each empty position on the board
		self.verbose = verbose

	def reset_history(self):
		self.state_history = []

	def take_action(self, env):
		# choose an action based on Epsilon-Greedy Strategy
		r = np.random.random()
		best_state = None
		if r < self.eps:
			# take a random action
			if self.verbose:
				print('Taking a random action')

			possible_moves = []
			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i, j):
						possible_moves.append((i, j))
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]
		else:
			# choose the best action based on current values of states
			# loop through all possible moves, get their values
			# keep track of the best value
			pos2value = {} # for debugging
			next_move = None
			best_value = -1
			for i in range(LENGTH):
				for j in range(LENGTH):
					if env.is_empty(i, j):
						# what is the state if we made this move?
						env.board[i, j] = self.sym
						state = env.get_state()
						env.board[i, j] = env.empty # don't forget to change it back!
						pos2value[(i, j)] = self.V[state]
						if self.V[state] > best_value:
							best_value = self.V[state]
							best_state = state
							next_move = (i, j)

			# if verbose, draw the board with the values
			if self.verbose:
				print('Taking a greedy action')
				for i in range(LENGTH):
					print('-------------------')
					row = ''
					for j in range(LENGTH):
						row += '|'
						if env.is_empty(i, j):
							row += '%.3f' % pos2value[(i, j)]
						else:
							if env.board[i, j] == env.x:
								row += '  x  '
							elif env.board[i, j] == env.o:
								row += '  o  '
					row += '|'
					print(row)
				print('-------------------')

		# make the move
		env.board[next_move[0], next_move[1]] = self.sym

	def update_state_history(self, s):
		# cannot put this in take_action, because take_action only happens
		# once every other iteration for each player
		# state history needs to be updated every iteration
		# s = env.get_state() # don't want to do this twice so pass it in
		self.state_history.append(s)

	def update(self, env):
		# we want to BACKTRACK over the states, so that:
		# V(prev_state) = V(prev_state) + alpha * (V(next_state) - V(prev_state))
		# where V(next_state) = reward if it's the most current state
		#
		# NOTE: we ONLY do this update at the end of an episode, i.e. when the game is over
		# not so for all the Reinforcement Learning algorithms
		reward = env.reward(self.sym)
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha * (target - self.V[prev])
			self.V[prev] = value
			target = value
		self.reset_history()


class Human:
	def __init__(self):
		pass

	def set_symbol(self, sym):
		self.sym = sym # env.x or env.o

	def take_action(self, env):
		while True:
			# break if we make a legal move
			move = input('Enter coordinates i, j for your move (i,j = 0, 1, 2):')
			if ',' not in move:
				print('Use a comma to seperate i and j')
				continue

			res = move.split(',')
			if len(res) != 2:
				print('Use the input format: i,j')
				continue

			i, j = res[0], res[1]
			i, j = i.strip(), j.strip()
			if not i.isdigit() or not j.isdigit():
				print('i and j must be integer')
				continue

			i, j = int(i), int(j)
			if i < 0 or i > 2 or j < 0 or j > 2:
				print('i and j must be in range [0, 2]')
				continue
			
			if env.is_empty(i, j):
				env.board[i, j] = self.sym
				break

	def update_state_history(self, s):
		pass

	def update(self, env):
		pass


# recursive function that will return all
# possible states (as ints) and who the corresponding winner is for those states (if any)
# (i, j) refers to the next cell on the board to permute (we need to try empty, x, o)
# impossible game states are ignored, i.e. 3 x's and 3 o's in a row simultaneously
# since that will never happen in a real game
def get_state_hash_and_winner(env, i=0, j=0):
	results = []

	for v in (env.empty, env.x, env.o):
		env.board[i, j] = v # if empty board it should already be 0 (empty)
		if j == LENGTH - 1:
			# j goes back to 0, increase i, unless i == LENGTH - 1, then we are done
			if i == LENGTH - 1:
				# the board is full, collect results and return
				state = env.get_state()
				ended = env.game_over(force_recalculate=True)
				winner = env.winner
				results.append((state, winner, ended))
			else:
				results += get_state_hash_and_winner(env, i + 1, 0)
		else:
			# increase j, i remains the same
			results += get_state_hash_and_winner(env, i, j + 1)

	return results


def initial_value(env, sym, state_winner_triples):
	# initialize state values as follows
	# if sym wins, V(s) = 1
	# if sym loses or draw, V(s) = 0
	# otherwise, V(s) = 0.5
	V = np.zeros(env.num_states)
	for state, winner, ended in state_winner_triples:
		if ended:
			if winner == sym:
				v = 1
			else:
				v = 0
		else:
			v = 0.5
		V[state] = v
	return V


def play_game(p1, p2, env, draw=False):
	# loops until the game is over
	current_player = None
	while not env.game_over():
		# alternate between players
		# p1 always starts first
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1

		# draw the board before the user who wants to see it makes a move
		if draw:
			if draw == 1 and current_player == p1:
				env.draw_board()
			if draw == 2 and current_player == p2:
				env.draw_board()

		# current player makes a move
		current_player.take_action(env)

		# update state histories
		state = env.get_state()
		p1.update_state_history(state)
		p2.update_state_history(state)

	if draw:
		env.draw_board()

	# do the value function update
	p1.update(env)
	p2.update(env)

	return env.winner


def main():
	# train the agent
	p1 = Agent()
	p2 = Agent()

	# set initial V for p1 and p2
	env = Environment()
	state_winner_triples = get_state_hash_and_winner(env)

	Vx = initial_value(env, env.x, state_winner_triples)
	p1.setV(Vx)

	Vo = initial_value(env, env.o, state_winner_triples)
	p2.setV(Vo)

	# give each player their symbol
	p1.set_symbol(env.x)
	p2.set_symbol(env.o)

	episodes = 10000
	for episode in range(episodes):
		if episode % 200 == 0:
			print('Episode %d finished!' % episode)
		play_game(p1, p2, Environment())

	# play human vs. agent
	human = Human()
	human.set_symbol(env.o)
	p1.set_verbose(True)
	p1.eps = 0
	while True:
		winner = play_game(p1, human, Environment(), draw=2)
		# make the agent player 1 because I want to see if it would
		# select the center as its starting move. If you want the agent
		# to go second you can switch the human and AI.
		if winner is None:
			print('It is a draw!')
		elif winner == human.sym:
			print('You win!')
		else:
			print('You lose!')
		answer = input('Play again? [Y/N]:')
		if answer[0] in ('n', 'N'):
			break


if __name__ == '__main__':
	main()
