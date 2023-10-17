from datetime import datetime, timezone
import json
import os.path

from tqdm import tqdm

base_path = "/data8T_1/qijiong/Data/Goodreads"
interaction_path = os.path.join(base_path, "goodreads_interactions_dedup.json")
book_path = os.path.join(base_path, "goodreads_book_works.json")

converted_interaction_path = os.path.join(base_path, "inter.csv")


def convert_to_timestamp(date_string):
    # 定义字符串的日期格式
    date_format = "%a %b %d %H:%M:%S %z %Y"
    # 将字符串转换为datetime对象
    dt = datetime.strptime(date_string, date_format)
    # 将datetime对象转换为timestamp（秒数）
    timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
    return timestamp


with open(interaction_path, 'r') as f_inter:
    with open(converted_interaction_path, 'w') as f_convert:
        for line in tqdm(f_inter):
            if line.endswith('\n'):
                line = line[:-1]
            data = json.loads(line)
            user_id, book_id, rating, timestamp = data['user_id'], data['book_id'], data['rating'], data['date_added']
            # a sample of timestamp: Fri Dec 09 15:51:44 -0800 2016
            timestamp = convert_to_timestamp(timestamp)

            f_convert.write(f"{user_id},{book_id},{rating},{timestamp}\n")
