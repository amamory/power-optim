# Standard library imports
import sys
import os
import argparse
import time
import subprocess
import atexit

def main():
    """ Run Minizinc model and plot the resulting schedule 

    """

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Minizinc model.'
                        )
    parser.add_argument('data', type=str, 
                        help='Minizinc parameters file.'
                        )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print ("ERROR: File %s not found" % (args.model))
        exit(1)
    if not os.path.isfile(args.data):
        print ("ERROR: File %s not found" % (args.data))
        exit(1)


    path_model=args.model
    path_data=args.data

    out = None

    print ("MODEL:", path_model)
    print ("DATA:", path_data)
    try:
        out = subprocess.Popen(["minizinc", "-m", path_model, "-d", path_data,
            "--output-mode", "json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE )
    except subprocess.CalledProcessError as e:
        print(e.output)
        print ("ERROR: could not call minizinc")
        sys.exit(1)

    # CONVERT THE MINIZINC OUTPUT INTO JSON 
    
    # the minizinc output is a list of bytes
    out_str = out.stdout.readlines()
    # print (type(out_str))
    # print (len(out_str))
    # print (type(out_str[0]))
    # the 2 last lines are useless
    out_str = out_str[:-2]
    # print (len(out_str))
    # print (out_str)
    # this will convert a list of bytes into a single string
    out_json=b''.join(out_str).decode('utf-8')
    # print (type(out_json))
    # out_json=out_json.replace('\\n', '\n')
    print("FORMATTED:")
    print(out_json)


if __name__ == "__main__":
    main()
