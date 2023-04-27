import json

from UniTok import UniDep

path = 'data/MIND-small-v2/{mode}-fake-v3'

modes = ['train', 'dev', 'test']

allowed_users = set(json.load(open('data/MIND-small-v2/allowed_user.json')))

for mode in modes:
    depot = UniDep(path.format(mode=mode))
    depot.filter(lambda x: x in allowed_users, col='uid')
    depot.export(path.format(mode=mode) + '-allowed')

