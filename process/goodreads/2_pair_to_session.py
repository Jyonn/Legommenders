import os

from tqdm import tqdm

base_path = "/data8T_1/qijiong/Data/Goodreads"

inter_path = os.path.join(base_path, "inter.csv")
session_path = os.path.join(base_path, "session.csv")

current_user = None
current_session = None
current_neg = None


def handle_session(session: list, neg: list) -> (str, str):
    ordered_session = []
    ordered_neg = []
    for book_id, timestamp in sorted(session, key=lambda x: x[1]):
        ordered_session.append(book_id)
    for book_id in neg:
        ordered_neg.append(book_id)
    return ' '.join(ordered_session), ' '.join(ordered_neg)


with open(inter_path, 'r') as f_inter:
    with open(session_path, 'w') as f_session:
        for line in tqdm(f_inter):
            if line.endswith('\n'):
                line = line[:-1]
            user_id, book_id, rating, timestamp = line.split(',')
            rating = int(rating)
            if rating == 3:
                continue
            pos = rating > 3

            if user_id != current_user:
                if current_user is not None:
                    ordered_session, ordered_neg = handle_session(current_session, current_neg)
                    if ordered_session:
                        f_session.write(f"{current_user}\t{ordered_session}\t{ordered_neg}\n")

                current_user = user_id
                current_session = []
                current_neg = []

            if pos:
                current_session.append((book_id, timestamp))
            else:
                current_neg.append(book_id)

        if current_session:
            ordered_session, ordered_neg = handle_session(current_session, current_neg)
            f_session.write(f"{current_user}\t{ordered_session}\t{ordered_neg}\n")
