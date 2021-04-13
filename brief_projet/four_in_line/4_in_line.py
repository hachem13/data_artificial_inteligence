import numpy as np 
import random

class Connect4():

    def __init__(self):
        self.rowCount = 6
        self.columnCount = 7
        self.board =  np.zeros((self.rowCount, self.columnCount))
        self.win = None
        self.game_over = False
        
    def isValidLocation(self, col):
        return self.board[self.rowCount-1][col] == 0  
    
    def getNextOpenRow(self, col):
        for r in range(self.rowCount):
            if self.board[r][col] == 0:
                return r
    def printBoard(self):
        # print board
         print(np.flip(self.board, 0))
              
    def dropPiece(self, row, col, piece):
        self.board[row][col] = piece
        
    def winningMove(self, piece):
        
        # check horizontal
        for c in range(self.columnCount-3):
            for r in range(self.rowCount):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True
                
        # check verticale
        for c in range(self.columnCount):
            for r in range(self.rowCount-3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True     
                
        # check positivly diagnols
        for c in range(self.columnCount-3):
            for r in range(self.rowCount-3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True  
        
        # check negativly diaghnols
        for c in range(self.columnCount-3):
            for r in range(3, self.rowCount):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True
            
    def choicePlay(self):
        choice = int(input('Choice your game: \n 1: Player1 Vs Player2 \n 2: Player1 VS Bot \n 3: Bot VS Bot \n'))
        return choice

    def gameOver(self):
        choice = self.choicePlay()
        if choice == 1: 
            turn = 0
            while not self.game_over:
            
                if turn == 0:
                    col = int(input('Player 1 make your selection (0-6):'))    
                    if self.isValidLocation(col): 
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, 1)
                        
                        if self.winningMove(1):
                            print('Player 1 wins !!!!')
                            self.game_over = True
                    
                else: 
                    col = int(input('Player 2 make your selection (0-6):'))
                        
                    if self.isValidLocation(col):
                        
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, -1)
                        
                        if self.winningMove(-1):
                            print('Player 2 wins !!!!')
                            self.game_over = True
                            
                self.printBoard()
                          
                turn += 1
                turn = turn % 2
        elif choice == 2: 
            turn = 0
            while not self.game_over:
                if turn == 0:
                    col = int(input('Player 1 make your selection (0-6):'))
                    if self.isValidLocation(col):
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, 1)
                        
                        if self.winningMove(1):
                            print('Player 1 wins !!!!')
                            self.game_over = True
                        
                else:  
                    col = random.randint(0, 6)
                    if self.isValidLocation(col):
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, -1)
                        
                        if self.winningMove(-1):
                            print(' Bot wins !!!!')
                            self.game_over = True
                            
                self.printBoard()
                          
                turn += 1
                turn = turn % 2
        else: 
            turn = 0
            while not self.game_over:
                if turn == 0:
                    col = random.randint(0, 6)
                    if self.isValidLocation(col):
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, 1)
                        
                        if self.winningMove(1):
                            print('Bot 1 wins !!!!')
                            self.game_over = True
                else:  
                    col = random.randint(0, 6)
                    if self.isValidLocation(col):
                        row = self.getNextOpenRow(col)
                        self.dropPiece(row, col, -1)
                        
                        if self.winningMove(-1):
                            print('Bot 2 wins !!!!')
                            self.game_over = True
                            
                self.printBoard()
                          
                turn += 1
                turn = turn % 2        
                
                
connect = Connect4()
connect.gameOver()

if __name__ == "__main__":
    Connect4()

