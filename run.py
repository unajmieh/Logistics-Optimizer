import subprocess  
import argparse  
import os  

def run_script(script, *args):  
    try:  
        print(f"Running: {script} with arguments: {args}")  
        subprocess.run(['python', script] + list(args), check=True)  
    except subprocess.CalledProcessError as error:  
        print(f"Error occurred while running {script}: {error}")  
        exit(1)  # Exit if any script fails.  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--nests', action='store_true', help='Generate nests using MainNestGeneration.py')  
    parser.add_argument('--variables', action='store_true', help='Run MainVariablesFile.py')  
    args = parser.parse_args()  

    if args.nests:  
        run_script('MainNestGeneration.py')  

    if args.variables:  
        run_script('MainVariablesFile.py')