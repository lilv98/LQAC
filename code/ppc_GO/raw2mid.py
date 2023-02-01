import pdb
import pandas as pd
import numpy as np
import re
import pickle

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_file():
    tbox = []
    abox_ec = []
    abox_ee = []
    with open('../raw/ontology.txt') as f:
        for line in f:
            line = line.strip('\n')
            if 'interacts' in line:
                e_1, e_2 = re.match(r'ObjectIntersectionOf\(<http://(.*)> ObjectComplementOf\(ObjectSomeValuesFrom\(<http://interacts> <http://(.*)>\)\)\)', line, re.M|re.I).groups()
                abox_ee.append([e_1, 'interactWith', e_2])
            elif 'hasFunction' in line:
                e, c = re.match(r'ObjectIntersectionOf\(<http://(.*)> ObjectComplementOf\(ObjectSomeValuesFrom\(<http://hasFunction> (.*)\)\)\)', line, re.M|re.I).groups()
                abox_ec.append([e, 'hasPhenotype', c])
            else:
                name_match = re.match(r'ObjectIntersectionOf\(<(.*)> ObjectComplementOf\(<(.*)>\)\)', line, re.M|re.I)
                if name_match:
                    matched = name_match.groups()
                    tbox.append([f'<{matched[0]}>', f'<{matched[1]}>'])
    
    with open('../raw/abox_ee_valid.txt') as f:
        for line in f:
            e_1, e_2 = line.strip().split('\t')
            abox_ee.append([e_1, 'interactWith', e_2])

    with open('../raw/abox_ee_test.txt') as f:
        for line in f:
            e_1, e_2 = line.strip().split('\t')
            abox_ee.append([e_1, 'interactWith', e_2])
    
    tbox = pd.DataFrame(tbox, columns=['h', 't']).drop_duplicates()
    abox_ec = pd.DataFrame(abox_ec, columns=['h', 'r', 't']).drop_duplicates()
    abox_ee = pd.DataFrame(abox_ee, columns=['h', 'r', 't']).drop_duplicates()

    return tbox, abox_ec, abox_ee

def transductor(ot_data):
    ret = []
    ot_h = set(ot_data['h'].unique().tolist())
    ot_t = set(ot_data['t'].unique().tolist())
    ot_bridge = ot_h & ot_t
    ot_dict = ot_data.groupby('h')['t'].apply(lambda x: x.tolist()).to_dict()
    ot_data_list = ot_data.values.tolist()
    for h, t in ot_data_list:
        ret.append([h, t])
        if t in ot_bridge:
            for tt in ot_dict[t]:
                ret.append([h, tt])
    return pd.DataFrame(ret, columns=['h', 't']).drop_duplicates()

def transductor_master(ot_data):
    mid = transductor(ot_data)
    length = len(mid)
    for i in range(10):
        mid = transductor(mid)
        if len(mid) != length:
            length = len(mid)
        else:
            return mid

def entity_filter(kg_data, is_data, k):
    entity_frequency = pd.concat([kg_data['h'], kg_data['t']]).value_counts()
    entity_frequency = entity_frequency.reset_index()
    entity_frequency.columns = ['e', 'count']
    valid_entities = entity_frequency[entity_frequency['count'] >= k]['e']
    kg_data = kg_data[kg_data['h'].isin(valid_entities) & kg_data['t'].isin(valid_entities)]
    is_data = is_data[is_data['h'].isin(valid_entities)]
    return valid_entities, kg_data.reset_index(drop=True), is_data

def split(data, train_ratio=0.95):
    mask = np.random.rand(len(data)) < train_ratio
    train = data[mask].reset_index(drop=True)
    return train

if __name__ == '__main__':
    ot_data, abox_ec, abox_ee = read_file()
    # ot_data = transductor_master(tbox).reset_index(drop=True)
    valid_entities, kg_data, is_data = entity_filter(abox_ee, abox_ec, k=1)
    is_data = is_data[['h', 't']].reset_index(drop=True)

    save_root = '../mid/'
    kg_data.to_csv(save_root + 'kg_data_all.csv')
    ot_data.to_csv(save_root + 'ot.csv')
    is_data.to_csv(save_root + 'is_data_all.csv')
    valid_entities.to_csv(save_root + 'e.csv')

    kg_data_all = kg_data
    is_data_all = is_data
    ot_data = ot_data
    print('Done reading data.')
    kg_data_train = split(kg_data_all)
    is_data_train = split(is_data_all)
    kg_data_train.to_csv(save_root + 'kg_data_train.csv')
    is_data_train.to_csv(save_root + 'is_data_train.csv')
    print('Done splitting data.')
    kg_dict_all = kg_data_all.groupby(['h', 'r'])['t'].apply(lambda x: x.tolist()).to_dict()
    is_dict_all = is_data_all.groupby('h')['t'].apply(lambda x: x.tolist()).to_dict()
    kg_dict_train = kg_data_train.groupby(['h', 'r'])['t'].apply(lambda x: x.tolist()).to_dict()
    is_dict_train = is_data_train.groupby('h')['t'].apply(lambda x: x.tolist()).to_dict()
    print('Done getting mapper.')

    save_obj(kg_dict_all, save_root + 'kg_dict_all.pkl')
    save_obj(is_dict_all, save_root + 'is_dict_all.pkl')
    save_obj(kg_dict_train, save_root + 'kg_dict_train.pkl')
    save_obj(is_dict_train, save_root + 'is_dict_train.pkl')
