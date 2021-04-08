import numpy as np
def idx_to_pos(idx, dl, origin):
    return idx*dl + origin

def pos_to_idx(pos, dl, origin):
    return (pos - origin) / dl

def main():
    origin = np.array([-1.5,1,0.5])
    dl = 0.5
    pos1 = np.array([12.45,9.7,-0.3])
    idx1 = pos_to_idx(pos1, dl, origin)

    pos2 = idx_to_pos(idx1, dl, origin)
    print(pos1)
    print(idx1)
    print(pos2)
if __name__ == '__main__':
    main()
    