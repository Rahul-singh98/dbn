# for testing 
# from test import testing

# for live predictions
from run import testing 
import argparse
import time

def run():
    parser = argparse.ArgumentParser(description='Deep Belief Network')
    parser.add_argument('-c','--csv_filepath' ,
                         default='./data/NIFTY1.csv' ,
                         type=str ,
                         help='path of csv file to train')
    parser.add_argument('-m' ,'--saved_model_path' ,
                        default='./models/15minregressor.pkl' , 
                        type=str ,
                        help='Path of the saved pickle model')
    parser.add_argument('-i','--save_output_image' ,
                         default=0 ,
                         type=int ,
                         required=False, 
                         help='to save the test vs prediction graph')

    args = parser.parse_args()

    if args.save_output_image == 0:
        testing(args.csv_filepath,  save_output_image=False , model_path=args.saved_model_path )

    else:
        testing(args.csv_filepath,  save_output_image=True , model_path=args.saved_model_path )

if __name__ == "__main__":
    try:
        while True:
            run()
            time.sleep(3)

    except KeyboardInterrupt:
        print('\nExit')