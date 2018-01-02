"""
I don't know why using pbtxt format, but I am gonna convert it into json
"""
import os
import json
from pprint import pprint
import sys


def parse_pb_text(pb_f):
    if os.path.exists(pb_f):
        with open(pb_f, 'r') as f:

            is_start = False
            is_end = False

            all_items = []
            one_dict = dict()
            for l in f.readlines():
                l_s = l.strip()

                if l_s != "":
                    if l_s.endswith('{'):
                        is_start = True
                    elif l_s.endswith('}'):
                        is_end = True
                    else:
                        is_start = False
                        is_end = False

                    if not is_start and not is_end:
                        one_dict[l_s.split(':')[0]] = l_s.split(':')[1].strip(). \
                            strip('"').strip('"').strip("'").strip("'")
                    else:
                        if len(one_dict.keys()) >= 1:
                            all_items.append(one_dict)
                        one_dict = dict()
            pprint(all_items)
        to_save_f = os.path.join(os.path.dirname(pb_f), os.path.basename(pb_f).split('.')[0] + '.json')
        save_json_to_local(to_save_f, all_items)


def save_json_to_local(to_save_file, json_obj):
    with open(to_save_file, 'w+') as f:
        json.dump(json_obj, f, indent=4, ensure_ascii=False)
    print('json has been saved into {}'.format(to_save_file))
    print('Done!')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('specific a labelmap.pbtxt file please')
    else:
        parse_pb_text(sys.argv[1])
