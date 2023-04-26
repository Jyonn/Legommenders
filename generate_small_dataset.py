import json

from UniTok import UniDep

path = 'data/MIND-small-v2/{mode}-fake'

modes = ['train', 'dev', 'test']

allowed_users = json.load(open('data/MIND-small-v2/allowed_user.json'))

for mode in modes:
    depot = UniDep(path.format(mode=mode))
    depot.filter(lambda x: x in allowed_users, col=depot.id_col)
    depot.export(path.format(mode=mode) + '-allowed')

