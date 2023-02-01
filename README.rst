Features of the fork:


- Uses the Glicko rating system instead of the Elo rating system 
- Supports a draw button 
- New -d argument to find duplicates. Using -d will show the user two similar photos and asks the user to either keep both, or delete one of them. The deleted photos will be moved to a "duplicates" folder

.. code-block:: bash
usage: drank.py [-h] [-r] [-d] [-ro N_ROUNDS] [-fi FIGSIZE FIGSIZE] photo_dir

Uses the Elo ranking algorithm to sort your images by rank. The program globs for .jpg images to present to you in
random order, then you select the better photo. After n-rounds, the results are reported. Click on the "Select" button
or press the LEFT or RIGHT arrow to pick the better photo.

positional arguments:
  photo_dir             The photo directory to scan for .jpg images

optional arguments:
  -h, --help            show this help message and exit
  -r, --rank            rank images
  -d, --duplicates      display duplicates
  -ro N_ROUNDS, --n-rounds N_ROUNDS
                        Specifies the number of rounds to pass through the photo set (3)
  -fi FIGSIZE FIGSIZE, --figsize FIGSIZE FIGSIZE
                        Specifies width and height of the Matplotlib figsize (20, 12)
