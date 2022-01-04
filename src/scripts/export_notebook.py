if __name__ == '__main__':
    import os
    from subprocess import call

    project_dir = os.getcwd().split('src')[0]
    call(['python', '-m', 'nbconvert', project_dir+'final-project.ipynb', '--to html'])