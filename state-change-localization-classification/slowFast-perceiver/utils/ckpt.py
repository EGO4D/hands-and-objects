import copy
import re


def convert(name):
    if 'res' in name:
        m = re.match('(\w+)\.(\d)\.(\w+)\.(\d).(\w+)\.(\d)\.(\w+)(\d)_conv(.+)', name)
        if m is not None:
            n1 = int(m.group(2))
            n2 = int(m.group(4))
            n3 = int(m.group(6))
            n4 = int(m.group(8))
            suffix = m.group(9)

            out = r's{}.pathway{}_res{}.branch{}{}'.format(n1 + 1, n2, n3, n4, suffix)
            return out
        m = re.match('(\w+)\.(\d)\.(\w+)\.(\d).(\w+)\.(\d)\.(\w+)(\d)_norm(.+)', name)
        if m is not None:
            n1 = int(m.group(2))
            n2 = int(m.group(4))
            n3 = int(m.group(6))
            n4 = int(m.group(8))
            suffix = m.group(9)

            out = r's{}.pathway{}_res{}.branch{}_bn{}'.format(n1 + 1, n2, n3, n4, suffix)
            if 'norm' in out:
                out = out.replace('norm', 'bn')
            return out
        m = re.match('(\w+)\.(\d)\.(\w+)\.(\d).(\w+)\.(\d)\.(\w+)(\d)\.conv_(\w)(.+)', name)
        if m is not None:
            n1 = int(m.group(2))
            n2 = int(m.group(4))
            n3 = int(m.group(6))
            n4 = int(m.group(8))
            n5 = m.group(9)
            suffix = m.group(10)

            out = r's{}.pathway{}_res{}.branch{}.{}{}'.format(n1 + 1, n2, n3, n4, n5, suffix)
            return out
        m = re.match('(\w+)\.(\d)\.(\w+)\.(\d).(\w+)\.(\d)\.(\w+)(\d)\.norm_(\w)(.+)', name)
        if m is not None:
            n1 = int(m.group(2))
            n2 = int(m.group(4))
            n3 = int(m.group(6))
            n4 = int(m.group(8))
            n5 = m.group(9)
            suffix = m.group(10)

            out = r's{}.pathway{}_res{}.branch{}.{}_bn{}'.format(n1 + 1, n2, n3, n4, n5, suffix)
            return out


    elif 'multipathway_blocks' in name:
        m = re.match('(\w+)\.(\d)\.(\w+)\.(\d)(.+)', name)

        n1 = int(m.group(2))
        n2 = int(m.group(4))
        suffix = m.group(5)

        # out = re.sub('(\w+)\.(\d)\.(\w+)\.(\d)(.+)', r's{}.pathway{}_stem{}'.format(n1 + 1, n2, suffix), name)
        out = r's{}.pathway{}_stem{}'.format(n1 + 1, n2, suffix)
        if 'norm' in out:
            out = out.replace('norm', 'bn')
        return out
    elif 'multipathway_fusion' in name:
        m = re.match('(\w+)\.(\d)\.(\w+)(.+)', name)

        n1 = int(m.group(2))
        suffix = m.group(4)

        # out = re.sub('(\w+)\.(\d)\.(\w+)\.(\d)(.)', , name)
        out = r's{}_fuse{}'.format(n1 + 1, suffix)
        out = out.replace('conv_fast_to_slow', 'conv_f2s')
        if 'norm' in out:
            out = out.replace('norm', 'bn')
        return out

    elif 'proj' in name:
        _, suffix = name.rsplit('.', 1)
        return 'head.projection.' + suffix

    else:
        return name


def convert_state_dict(state_dict):
    dict_cpy = copy.deepcopy(state_dict)
    for param in dict_cpy:
        state_dict[convert(param)] = state_dict.pop(param)
    return state_dict
