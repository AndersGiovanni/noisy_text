import os

minN = 1
maxN = 7

levels = ['charN', 'wpieceN', 'wordN']
if not os.path.isdir('scores'):
    os.mkdir('scores')

for end in range(minN,maxN + 1):
    for beg in range(minN, end+1):
        for level in levels:
            cmd = 'python3 classify.py --' + level + ' ' + str(beg) + '-' + str(end)
            for otherLevel in levels:
                if otherLevel != level:
                    cmd += ' --' + otherLevel + ' 0'
            cmd += ' > scores/' + level + '.' + str(beg) + '-' + str(end)
            print(cmd)
