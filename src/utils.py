import os, shutil, re, sys
import pandas as pd
import numpy as np
import psutil

if not sys.platform.startswith('win'):
    import resource

# 检测运行环境
def in_notebook():
    return 'IPKernelApp' in getattr(globals().get('get_ipython', lambda: None)(), 'config', {})


def show_dataframe(data):
    if in_notebook():
        from IPython.display import display
        display(data)
    else:
        import tabulate
        print(tabulate.tabulate(data, headers='keys', tablefmt='mixed_grid'))

# 防止多进程乱序执行时 即使判断不存在路径，仍然创建失败
def try_make_dir(path):
    if not os.path.exists(path) and ('place_holder' not in path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)

# 写入基因名到文件
def write_genes_to_file(genes_series, file_path):
    if not isinstance(genes_series, pd.Series):
        genes_series = pd.Series(list(genes_series))
    genes_series.to_csv(file_path, index=False, header=False)

# 从文件读取基因名到列表
def read_genes_from_file(file_path):
    try:
        # 读取文件内容
        genes_series = pd.read_csv(file_path, header=None).squeeze()
        # 确保结果是一个Series对象
        if isinstance(genes_series, pd.Series):
            return genes_series.tolist()
        else:
            # 如果只有一个元素，将其转化为列表
            return [genes_series]
    except Exception as e:
        return []

def get_memory_usage_detail(mark_info=None):
    if sys.platform.startswith('win'):
        return get_memory_usage_win(mark_info=mark_info)
    else:
        return get_memory_usage_linux(mark_info=mark_info)

def get_memory_usage_win(mark_info=None):
    process = psutil.Process()
    memory_info = process.memory_full_info()
    memory_rss = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    memory_vms = memory_info.vms / (1024 * 1024)  # Convert bytes to MB
    memory_uss = memory_info.uss / (1024 * 1024)  # Unique Set Size

    if mark_info is not None:
        print(f"Memory Usage -> {mark_info}")
        print(f"\tRSS (Resident Set Size) : {memory_rss:.2f} MB")
        print(f"\tVMS (Virtual Memory Size) : {memory_vms:.2f} MB")
        print(f"\tUSS (Unique Set Size): {memory_uss:.2f} MB")
        print(f"\tMax RSS : - MB")
    return memory_rss, memory_vms, memory_uss, None

def get_memory_usage_linux(mark_info=None):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_rss = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    memory_vms = memory_info.vms / (1024 * 1024)  # Convert bytes to MB
    memory_uss = process.memory_full_info().uss / (1024 * 1024)  # Unique Set Size

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = usage.ru_maxrss / 1024  # Convert from KB to MB
    
    if mark_info is not None:
        print(f"Memory Usage -> {mark_info}")
        print(f"\tRSS (Resident Set Size) : {memory_rss:.2f} MB")
        print(f"\tVMS (Virtual Memory Size) : {memory_vms:.2f} MB")
        print(f"\tUSS (Unique Set Size): {memory_uss:.2f} MB")
        print(f"\tMax RSS : {max_rss:.2f} MB")
    
    return memory_rss, memory_vms, memory_uss, max_rss

def split_step(total, step):
    """
    Splits the total iterable into chunks of size step.
    
    Args:
    total: The iterable to be split.
    step: The size of each chunk.
    
    Returns:
    A list of chunks, where each chunk is a sublist of the total.
    """
    result = []
    for i in range(0, len(total), step):
        result.append(total[i:i + step])
    return result


def split_range(total, step):
    """
    Split a total range into smaller ranges of a given step size.

    Parameters
    ----------
    total : int
        The total length of the range to be split.
    step : int
        The step size for splitting the range.

    Returns
    -------
    list of tuples
        A list of tuples, each representing the start and end of a sub-range.

    Examples
    --------
    >>> split_range(10, 3)
    [(0, 3), (3, 6), (6, 9), (9, 10)]
    """
    return [(i, min(i+step, total)) for i in range(total) if i%step==0 or i==total]


def split_batch(total, batch_num):
    """
    Split a list into a specified number of batches.

    Parameters
    ----------
    total : list
        The list to be split into batches.
    batch_num : int
        The number of batches to split the list into.

    Returns
    -------
    list of lists
        A list of lists, where each sublist is a batch of the original list.

    Examples
    --------
    >>> split_batch([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    """
    k, m = divmod(len(total), batch_num)
    return [total[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(batch_num)]

def get_onehot(df_in:pd.DataFrame, columns, dummy_na=False, keep_input=False, sep=None):
    '''
    describe
    -------
    将df中的指定列转换为独热编码后放回该列原位置\n

    param
    -------
    `df_in`: input DataFrame
    `columns`: 要转换的列名
    `dummy_na`: 是否返回一列作为缺失指示
        `False`: 将缺失值填充到所有生成列
        `True`: 单独生成缺失指示列,此时所有one-hot列均不存在缺失
    `keep_input`: 是否保留原来输入的列（可用于对照检查）
    `sep`: 对输入列进行切分的字符串。对“购物篮数据”尤其有效，可构造多标签哑变量，防止大量项集哑变量
        - `None`: 默认不切分
        - str: 输入任意字符串，将以该字符串对输入列进行切分`pd.Series.str.split(sep)`

    return
    -------
    `df` 转换后的DataFrame
    '''
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    df = df_in.copy()
    columns = [columns] if type(columns)==str else columns
    for on in columns:

        if len(df[on].dropna().unique()) <=2: # 若该列的非空唯一值种类小于2 则不必生成哑变量
            continue
        
        if sep:
            na_mask = df[on].isna()
            df[on] = df[on].astype(str).str.split(sep)
            df.loc[na_mask,on] = np.nan

        dic = df[[on]].to_dict('records')
        onehot = pd.DataFrame(data=vec.fit_transform(dic).toarray(),columns=vec.feature_names_, index=df.index)
        print(f'生成哑变量：{vec.feature_names_}')

        if df[on].isna().any():
            if dummy_na:
                # 若设置了dummy_na 则增加缺失指示列
                onehot[on].replace(np.nan,1,inplace=True)
                onehot = onehot.rename(columns={on:f'{on}_NA'})
            else:
                # 否则将缺失值填充到所有生成列
                onehot.loc[onehot[on].isna()]=np.nan
                onehot.drop(columns=on,inplace=True)
        else:
            print(f'"{on}"列无缺失值,无需生成缺失指示列')

        #获取输入列在原df中的位置，并将转换后的one-hot编码放回该位置
        loc_on = [ix for ix,x in enumerate(df.columns) if x==on]
        if len(loc_on)>1:
            raise ValueError(f'存在多个名为 "{on}" 的字段，请先使用uniColumns将df中的列名设为唯一')
        loc_on = loc_on[0]
        for n,col in enumerate(onehot.columns):
            df.insert(loc_on+n,col,onehot[col])

        df.drop(columns=on,inplace=True) if not keep_input else df # 删除原变量
    return df


def getfiles(path, *suffix, mode=0, pattern=None, get='full'):
    '''
    describe
    -------
    获取指定后缀和文件名模式的文件名单\n

    param
    -------
    path:       文件夹/路径\n
    suffix:     后缀，str\n
    mode:       
        - 0:仅搜索该路径
        - 1:递归搜索\n
    pattern:    正则表达式匹配文件名，包括文件后缀\n
    get:        
        - 'fn':仅返回文件名列表
        - 'full':返回整个路径列表\n

    return
    -------
    返回一个列表`list`包含所有符合查找条件的文件路径
    '''   

    #如果是文件则使用文件所在路径
    path = os.path.dirname(path) if os.path.isfile(path) else path

    #文件搜索器
    def searcher(path, mode, get):
        if mode ==0:
            for i in os.listdir(path):
                if os.path.isfile(f'{os.path.abspath(path)}/{i}'): #listdir会列出文件夹,需要判断是否为文件
                    yield f'{os.path.abspath(path)}/{i}' if get=='full' else i if get=='fn' else None
                else:
                    continue
        if mode ==1:
            for root,dirs,files in os.walk(path):
                for file in files:
                    yield f'{root}/{file}' if get=='full' else file if get=='fn' else None

    #在内部定义函数判断文件名是否符合规定pattern
    def jugePattern(pattern, txt):
        if pattern:
            return re.search(pattern,txt)
        if not pattern:
            return True

    #在内部定义函数判断文件后缀是否符合
    def jugeSuffix(suffix, txt):
        if suffix:
            suf = txt.split('.')[-1]
            suflist = list(map(lambda x: x.upper(),suffix)) + list(map(lambda x: x.lower(),suffix))
            return suf in suflist
        if not suffix:
            return True
    
    file_name_list = []
    for file in searcher(path, mode, get):
        filename = os.path.split(file)[1]
        if jugePattern(pattern,filename) and jugeSuffix(suffix,filename):
            file_name_list.append(file)
    return list(map(lambda x:re.sub(r'\\+|\/+','/',x),file_name_list))

