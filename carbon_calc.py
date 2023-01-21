carbon_per_second = 350 * 0.722 / 3600


data = """
		naml	lstur	nrms	dcn		din		bst
id		70s		73s		100s	2.3m	3.5m	3m
text	2.5m	3.5m	3.7m	3.2m	4.7m	4.5m
btok	2.5m	3.5m	3.7m	3.2m	4.7m	4.5m
plmnr	10.6m	12m		15m		0.5h	0.6h	16.5m
bdoc	80s		82s		2m		2.3m	3.5m	4m
prec	80s		82s		2m		2.3m	3.5m	4m

		naml	lstur	nrms	dcn		din		bst
id		17.5m	21m		28m		33m		55m		47m
text	38.5m	53m		1h		72m		74m		100m
btok	38.5m	53m		1h		72m		74m		100m
plmnr	2.5h	3h		4h		8h		10h		6.5h
bdoc	21m		24m		30m		38m		1h		1h
prec	21m		24m		30m		38m		1h		1h
"""

full_values = []

for line in data.split('\n'):
    numbers = line.split('\t')
    line_value = []
    for value in numbers:
        if not value:
            continue
        if value[0] not in '0123456789':
            continue
        if 'm' in value:
            value = value.replace('m', '')
            value = float(value) * 60
        elif 'h' in value:
            value = value.replace('h', '')
            value = float(value) * 60 * 60
        else:
            assert 's' in value
            value = value.replace('s', '')
            value = float(value)
        value *= carbon_per_second * 4
        # print value with 0 decimal places and occupy 8 spaces with no tab
        # print('{:.0f}'.format(value).rjust(8), end='\t')
        line_value.append(int(value))
    # print()
    assert len(line_value) == 6 or not line_value
    if line_value:
        full_values.append(line_value)

for i in range(len(full_values)):
    full_values[i] = [*full_values[i][:4], *full_values[i][5:], full_values[i][4]]

for i in range(6):
    for v in full_values[i]:
        print(f'& {v}', end=' ')
    print()
    for v in full_values[i+6]:
        print(f'& {v}', end=' ')
    print('\\\\')
