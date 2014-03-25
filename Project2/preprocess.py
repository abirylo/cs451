from sys import stdin, stdout
from string import split
from struct import pack
 
def main():
    numbers = (int(word) for line in stdin for word in split(line))
    for number in numbers:
        stdout.write(pack("H", number))
 
if __name__ == "__main__":
    main()
