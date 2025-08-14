# Chess piece classifier

This Python project was a class assignment to make a chess piece classifier by using pre-existing training data of over 6400 chessboard squares. I reduced and selected 10 features using Principal Component Analysis (PCA) then using K-Nearest Neighbor (KNN) as my classifier.

## Stats:
### Running evaluation with the clean data.
- Square mode: score = 98.2% correct
- Board mode: score = 98.2% correct
### Running evaluation with the noisy data.
- Square mode: score = 94.4% correct
- Board mode: score = 94.5% correct

Board mode means having contextual knowledge of the board e.g knowing which position the square is in within the 64 square board.

## Installation

1. Clone this repository to your local machine.
3. Install the required dependencies using `pip install -r requirements.txt`.
8. Run the classifer using `python evaluate.py`.
