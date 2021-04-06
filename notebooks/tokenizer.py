import re

def token_based_regex_processing(token):
    token = re.sub(r'^([0-9\.]+\/|\/[0-9\.]+)$', lambda match:        re.sub(r'(\/)', r'', match.group(0), flags=re.U), token, re.U)
    token = re.sub(r'([\\\/]$|^[\\\/])', r'', token, re.U)
    token = re.sub(r'(^[\.]+$)', r'', token, re.U)
    token = re.sub(r'^([0-9]+\.)$', lambda match:        re.sub(r'([0-9]+\.)', r'\g<1>0', match.group(0), re.U), token, re.U)
    token = re.sub(r'(^\.[0-9]+$)', lambda match:        re.sub(r'(\.[0-9]+)', r'0\1', match.group(0), re.U), token, re.U)

    token = re.sub(r'^([0-9\.]+\/[0-9\.]+)\/([0-9\.]+\/[0-9\.]+)$', r'\1 \2', token, re.U)
    
    "750.000 = 750"
    token = re.sub(r'^([0-9]+\.[0]+)$', lambda match: str(int(float(match.group(0)))), token, re.U)
    
    return token

def remove_duplicate_tokens(token_list):
    result_token_list = []
    
    for token in token_list:
        if token not in result_token_list         or re.search(r'^[0-9\.]+$', token) != None:
            result_token_list.append(token)
    
    return ' '.join(result_token_list)


''' if there is dot(.) in the string it is replaced with space '''
def replace_dot(sample):
    return re.sub(r'([^0-9\s]+\.[^0-9\s]*)', lambda match:        re.sub(r'([\.])', r' ', match.group(0), re.U) , sample, re.U)



'''
This function will apply some regex rules like remove unwanted caracters, reduce sequential spaces and
 will split sample into tokens by space.
'''
def tokenize(sample):
    
    #print('raw string:')
    #print(sample)
    
    #if type(sample) != unicode:
     #   sample = sample.decode('utf8')
        
    sample = sample.lower()
    sample = re.sub(r'([^0-9a-z\.\s\(\)\\\/\-\&#]+)', r'', sample, re.U)
#     sample = re.sub(r'([\&])', r' \1 ', sample, re.U)
    sample = re.sub(r'(\([^\(\)]+\))', lambda match:        re.sub(r'([\(\)]+)', r' ', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([#])', r' no ', sample, re.U)
    sample = re.sub(r'([0-9\.]+\-[0-9\.]+\-*[0-9\.]*)', lambda match:        re.sub(r'([\-])', r'/', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([a-z]+&[a-z]+)', lambda match:        re.sub(r'([&])', r' and ', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([\-])', r' ', sample, re.U)
    
    '''1x5 to 1/5'''
    sample = re.sub(r'([0-9]+[x][0-9]+)', lambda match:        re.sub(r'([x])', r'/', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([0-9\.]+[a-z]+)', lambda match:        re.sub(r'([0-9\.]+)', r'\1 ', match.group(0), re.U), sample, re.U)
    
    
    '''specific to package and volume'''
    sample = re.sub(r'(\s*[\\\/]\s*)', lambda match: match.group(0).strip(), sample, re.U)
    sample = re.sub(r'([0-9\.]+\/[a-z]+)', lambda match:        re.sub(r'([\/])', r'\/1 ', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([a-z]+\/|\/[a-z]+)', lambda match:        re.sub(r'([\/])', r' ', match.group(0), re.U), sample, re.U)
#     sample = re.sub(r'([0-9\.]+\&[0-9\.]+)', lambda match:\
#         re.sub(r'([\&])', r' ', match.group(0), re.U), sample, re.U)
#     sample = re.sub(r'([0-9\.\\\/a-z]+)', lambda match:\
#         re.sub(r'([0-9\.\\\/]+)', r' \1 ', match.group(0), re.U), sample, re.U)
    sample = re.sub(r'([a-z]+[0-9\.]+)', lambda match:        re.sub(r'([0-9\.]+)', r' \1', match.group(0), re.U), sample, re.U)
   
    sample = re.sub(r'(\')', r'', sample, re.U)
    sample = re.sub(r'([\\\/]+)', r'/', sample, re.U)
    
    '''12.13.5 = 12/23.5'''
    sample = re.sub(r'([0-9\.]+\.[0-9\.]+\.[0-9\.]+)', lambda match:        re.sub(r'([\.])', r'/', match.group(0), count=1, flags=re.U), sample, re.U)
    
    '''12/.75 = 12/0.75'''
    sample = re.sub(r'\/\.[0-9]+', lambda match:        re.sub(r'\/\.', r'/0.', match.group(0), re.U), sample, re.U)
    
    ''' ./20 = 20'''
    sample = re.sub(r'[^0-9]\.\/[0-9]', lambda match:        match.group(0).replace('/', ' '), sample, re.U)
    
    '''.1.0 = 1.0'''
    sample = re.sub(r'\.[0-9]+\.[0-9]+', lambda match:        match.group(0).lstrip('.'), sample, re.U)
    
    '''.750 = .75'''
    sample = re.sub(r'\.[1-9]+[0]+', lambda match:        match.group(0).rstrip('0'), sample, re.U)
    
    '''750.000 = 750'''
    sample = re.sub(r'([0-9]+\.[0]+)', lambda match: match.group(0).rstrip('0')[:-1], sample, re.U)

    
    '''4/6/12 = 4/6 12'''
    split_slash = lambda splits: '/'.join(splits[:2]) + ' ' + splits[2]
    sample = re.sub(r'[0-9]+/[0-9]+/[0-9\.]+', lambda match: split_slash(match.group(0).split('/')), sample, re.U)
    
    sample = replace_dot(sample)
    
    sample = re.sub(r'(\s+)', r' ', sample, re.U)
    
    #print('after regex processing:')
    #print(sample)
    
    token_list = [token.strip() for token in sample.strip().split(' ')]
    
    token_list = [token_based_regex_processing(token) for token in token_list]
    
    #print('after token based regex processing:')
    #print(token_list)
        
    return remove_duplicate_tokens(token_list)
#     return token_list

