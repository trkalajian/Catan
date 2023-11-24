import subprocess

def run_jar_with_input(jar_path, input_text):
    try:
        process = subprocess.Popen(['java', '-jar', jar_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        # Send input to the Java process
        process.stdin.write(input_text + '\n')
        process.stdin.flush()
        
        # Read output from the Java process
        output = process.stdout.readline().strip()
        
        # Wait for the process to finish and close it
        process.communicate()
        return output
    except FileNotFoundError:
        return "Java not found. Please make sure Java is installed and configured properly."

if __name__ == "__main__":
    jar_file_path = "JSettlers.jar"  # Replace with the path to your .jar file
    user_input = input("Enter input for the Java program: ")
    result = run_jar_with_input(jar_file_path, user_input)
    print("Output from Java program:", result)
