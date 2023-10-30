# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

""""
@brief Note template generation 
@file hello_world.py
@author Afonso Diela
@history 2023-10-30 | Creation
"""
import os
import sys
import argparse
import re

class Gen_Note():
    def __init__(self, topic_name):
        pass

def note_content_template(topic):
    """
    @brief Note contents
        # Title - Notes
        ## Table of Contents (ToC)
        ## Overview
        ## Applications
        ## Tools & FrameWorks
        ## Hello World!
        ## References
    """
    note_name_path = topic
    Title = topic.title()
    # Note Headlines
    line = ""
    # line += str("# Title - Notes")
    # line += "\n".join(line)
    line += str(f"# {Title} - Notes" + "\n")
    line += str(f"## Table of Contents (ToC)"+ "\n")
    line += str(f"## Overview"+ "\n")
    line += str(f"## Applications"+ "\n")
    line += str(f"## Tools & FrameWorks"+ "\n")
    line += str(f"## Hello World!"+ "\n")
    line += str(f"## References"+ "\n")

    # TODO: check if file name has space
    # note file path name
    note_file_name_path = note_name_path +"-"+"notes"+".md"

    with open(note_file_name_path, 'w') as note_file_name:
        note_file_name.write(line)
        # print(line, end='')

def create_repo(repo_name):
    """
    @brief create repo
        - repo_name: repo name (topic-name, docs, lab ...)
        - cd repo_name and create docs and lab folder

    """
    # repo_name = topic
    # os.mkdir(repo_name)
    return os.mkdir(str(repo_name))


def note_gen(topic_name):
    # create note repo
    repo_name = topic_name +"-"+"notes"
    create_repo(repo_name)
    # print("repo_name:", repo_name)
    # get full path of repo_name
    cwd = os.getcwd()
    # Get the path to a specific folder
    repo_name_path = os.path.join(cwd, repo_name)

    if repo_name_path:
        # cd dir
        os.chdir(repo_name_path)
        # create docs dir
        create_repo("docs")
        # create lab dir
        create_repo("lab")        
        # create note file template
        note_content_template(topic_name)

def gen_helper():
    print("Usage: python hello_world.py [options] [<NotePath>] [<NoteName>]\n")        
    print("[<NotePath>]")
    print("\tSpecify a path if you want to generate the note in a different location")
    print("[<NoteName>]")
    print("\tNote/Topic name")
    print("Options:")
    print("\t-v, --verbose: print debug informationn")
    print("\t-a, --all: display the list of current notes")
    print("\t-h, --help: see usage")

# main
if __name__ == '__main__':
    """Main function.

        Args:
            -v, --verbose: print debug information
            -a, --all: display the list of current notes
            -h, --help: see usage
    """ 
    # args options 
    options=['-a', '-h', '-v']
    
    if (len(sys.argv) >= 2):
        ## List created from absolute path of the file
        hello_world_name_path = __file__.split('/')
        # TODO: Use 'argparse' for args considerations
        # TODO: add argvs consideration function: check + actions
            # - if argv 1 is empty and different "-h" or "--help" => argv2 = argv1
            # and argv2 == argv2 last?
        
        # gen_options = str(sys.argv[1])
        # topic_path  = str(sys.argv[2])
        # topic_name  = str(sys.argv[3])
        
        # TODO: in helper() check => sys.argv[1]) is not options 
        """ IN THIS VERSION I'M ONLY TAKING A SINGLE ARGUMENT WHICH THE NAME OF 'TOPIC/NOTE' """
        if (str(sys.argv[1]) != None):
            topic_name = str(sys.argv[1])
            # generate note template contents
            note_gen(topic_name)
        # elif (str(sys.argv[1]) != None) and (str(sys.argv[2]) != None) and (str(sys.argv[3]) != None):
        #     topic_name = "".join([str(sys.argv[2]), str(sys.argv[3])])
        #     note_gen(topic_name)
        elif (str(sys.argv[1]) == None) and (str(sys.argv[2]) == None) and (str(sys.argv[3]) == None):
            topic_name = str(sys.argv[1])
            note_gen(topic_name)
        else:
            gen_helper()

        # print(f'hello_world_name_path: {hello_world_name_path}\nhello_world_name_path-len: {hello_world_name_path[len(hello_world_name_path) - 1]}')
        # print(f'topic_name: {topic_name} \ntopic_name-len: {len(topic_name)}')
        # exit()

    else:
        gen_helper()


