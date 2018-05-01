# -*- coding: utf-8 -*-
"""

Created by f.maire@qut.edu.au on Wed Jan 24 19:23:48 2018
Last modified Sun 11 March 2018


The aim of this assignment is to create a planner that given the state
of an assembly workbench can determine whether or not a target state 
is reacheable from the current state by a sequence of legal assembly actions.

The domain of the planner are objects constructed by connecting together 2D 
tetris pieces.

Here a tetris piece refers to any finite subset of occupied cells of a grid
that satisfies all the constraints listed below.
    We can associate a graph to represent a set of occupied cells.
    Each vertex of the graph corresponds to an occupied cell.
    Two vertices are connected by an edge if and only if their corresponding
    cells are adjacent (they share a cell-wall).
    
    We impose the following constraints on what we call "tetris pieces". 
    It is always the case that
        - the induced graph of the cells is connected
        - there is at least one cell in row 0 (top of the workbench)
        - there is at least one cell in column 0 (leftmost column of the grid)
        - each cell is labelled with the same integer id.
    
The tetris pieces are atomic. They cannot be split into simpler pieces.
However, we can connect them to make composite parts, and by connecting more
parts we can build complex objects.

Formally, a part is either a tetris piece or the result of moving one part
above another part and lowering the first part until it connects to the 
second part on at least one horizontal edge.

We will also consider a variation of the problem where during the assembly, 
we are also allowed to rotate a part 90 degree clockwise.

In other words, the construction process of a complex object is a 
sequence of actions.  The only allowed types of action are 
- lifting a part and dropping it onto another part
- rotating a part 90 degree clockwise (if allowed)

The state of the workbench is the set of parts present 
on the workbench.

We define a canonical order to list the parts present on a workbench.
A function is provide to put a state in this canonical order.



@author: frederic

"""


import numpy as np

import random

import generic_search


# ---------------------------------------------------------------------------

class TetrisPart(object):
    '''
    An instance of the class TetrisPart is 
    - either an atomic tetris piece. In this case, 'self.part_under' is None.
    - or a composite part obtained by dropping "self.part_above" onto 
      'self.part_under'
      
    *************************    WARNING   ************************* 
    If the dropped part 'part_above' does not get in contact with 'part_under'
    on at least one horizontal edge, then the resulting part is defective.
    This model the way Lego bricks connect.
    To indicate the failure, self.offset is set to None.
    
    For example dropping
        1 1
        0 1
        0 1
    onto 
        1 0
        1 0
        1 1
    with an offset of 1 results in
        1 1
        0 1
        0 1
    not
        1  1  1 
        1  0  1 
        1  1  1 
    
    But dropping 
        1 1 1
        0 0 1
    onto
        1 0 0
        1 1 1
    with an offset of 0 results in 
        1  1  1 
        1  0  1 
        1  1  1 
             
    'self.part' is a numpy array representing the part in a canonical way;
        self.part[0] is the row in contact with the workbench
        self.part[:,0] has at least one occupied cell.
        
    The method 'self.get_frozen()' returns the 'self.part' array as a tuple of
    tuple. This is useful when a immutable representation of the part is 
    required.
    
    The method 'self.rotate90()' rotates the part in place.
        
    '''

    def __init__(self, part_above, part_under=None, offset=0):
        '''
        The arguments 'part_above' and 'part_under' are TetrisPart's or their
        corresponding frozen representations.  This new part is obtained by 
        dropping 'part_above' onto 'part_under'.
        
        Take note of the ** WARNING ** in the header comment of this class.
        
        @param
          part_above : part that is dropped
          part_under : the other part
          offset : horizontal integer offset of 'part_above' when 
                dropped on to'part_under'. The offset is with respect
                to the configuration where the left sides of the two grids
                are aligned.
                
        @post
            if the drop is unsuccessful, that is if there is not horizontal
            edge contact when 'part_above' is dropped onto 'part_under, 
            then self.offset will be set to None to indicate construction
            failure.               
        '''
        if type(part_above) is TetrisPart:
            part_above = part_above.get_frozen()
        self.part_above = part_above
        if type(part_under) is TetrisPart:
            part_under = part_under.get_frozen()    
        self.part_under = part_under
        self.offset = offset
        self.frozen = None # frozen representation of this part
        assert not(self.part_above is None) # defensive programming
        self.make_part()
  
    
    def clone(self):
        '''
        Return a copy of itself
        '''
        tp = TetrisPart(None)
        tp.part_above = self.part_above
        tp.part_under = self.part_under
        tp.offset = self.offset
        tp.part = self.part.copy()
        tp.frozen = None #self.get_frozen()
        return tp
        
        
    def rotate90(self):
        '''
        Rotate in place this part 90 degrees clockwise.
        Warning: 
          Only self.part is rotated. The array 'self.part_above' and
          'self.part_under' are left unchanged.
        
        '''
        self.part = np.array(self.part[:,::-1]).transpose()
        self.frozen = None # mark as dirty 
                           # will be recomputed if needed in 'self.get_frozen()'
        
    
    def make_part(self):
        '''
            Compute the composite part
            obtained by dropping self.part_above onto self.part_under
            with an offset value 'self.offset'

            @pre
            'self.part_above' and 'self.part_under' are properly 
               constructed  parts.

            @post
                if the drop is unsuccessful, that is if there is no horizontal
                edge contact, then 'self.part' will be 'part_above' 
                and self.offset will be set to None.
                See class header comment.                
        '''
        if self.part_under is None:
            self.part = np.array(self.part_above)
            return
        offset = self.offset
        pa = np.array(self.part_above)
        pu = np.array(self.part_under)
        # compute the *inclusive ranges* of the horizontal coord x 
        # and vertical coord y
        # for the combined part obtained by stacking the bounding box
        # of 'pa' on top of the bounding box of 'pu'
        #    y_min = 0
        y_max = pa.shape[0]+pu.shape[0]-1
        # offset can be negative 
        x_min = min(0, offset) # pa x-coord can be negative
        x_max = max(pa.shape[1]-1+offset, pu.shape[1]-1)

        # align the parts 'pa' and 'pu' in a common grid 
        M_pu = np.zeros( (y_max+1,x_max-x_min+1), dtype=np.int )
        M_pa = M_pu.copy()
        if offset<0:
            # left side of pa at zero on the common matrix M
            # equivalently pu move right  -offset
            M_pu[:pu.shape[0],-offset:pu.shape[1]-offset] = pu
            M_pa[pu.shape[0]:, :pa.shape[1]] = pa  # pa just above pu
        else:
            # offset >= 0
            # left side of pu is at zero in the common matrix M
            M_pu[:pu.shape[0],:pu.shape[1]] = pu
            M_pa[pu.shape[0]:, offset:pa.shape[1]+offset] = pa
        
        # compute the y_drop
        M_pu = (M_pu!=0).astype(np.int) # 0,1 matrix
        C_pu = [ ''.join(str(v) for v in c) for c in M_pu.transpose()  ]
        D_pu = [ s.rfind('1') for s in C_pu]        
        # D_pu[i] is the y_coord of the top element in ith column of 'M_pu"

        drop_magic = 2*(pa.shape[0]+pu.shape[0])  
        # magic constant: sentinel value to detect
        # parts that do not connect properly
        
        # trick to detect non connecting parts
        D_pu = [ -drop_magic if v ==-1 else v for v in D_pu]        
                
        M_pa = (M_pa!=0).astype(np.int) # 0,1 matrix
        C_pa = [ ''.join(str(v) for v in c) for c in M_pa.transpose()  ]
        D_pa = [ s.find('1') for s in C_pa]
        D_pa = [ drop_magic if v==-1 else v for v in D_pa]
        # D_pa[i] is the y_coord of the bottom element in ith column of 'M_pa"
        
        drop = min(np.array(D_pa)-np.array(D_pu))-1  # drop should be non negative
        if drop>pu.shape[0]:
            # print('non connecting, drop = ', drop,pu.shape[0])
            self.part = np.array(self.part_above)
            self.offset = None # to indicate construction failure
            return
            
        # otherwise connection is fine      
        nr_new_part = max(
                y_max-drop+1, # y coord of top cell of pu is y_max-drop
                pu.shape[0]
                )
        
        M = np.zeros( (nr_new_part,x_max-x_min+1), dtype=np.int )
        M_pa = M.copy()
        M_pu = M.copy()
        
        try: # debugging leftover
            
            if offset<0:
                # left side of pa at zero on the common matrix M
                # equivalently pu move right  -offset
                M_pu[:pu.shape[0],-offset:pu.shape[1]-offset] = pu
                M_pa[pu.shape[0]-drop:pu.shape[0]-drop+pa.shape[0] , 
                     :pa.shape[1]] = pa  # pa just above pu
            else:
                # offset >= 0
                # left side of pu is at zero in the common matrix M
                M_pu[:pu.shape[0],:pu.shape[1]] = pu
                M_pa[pu.shape[0]-drop:pu.shape[0]-drop+pa.shape[0], 
                     offset:pa.shape[1]+offset] = pa
        except ValueError:
            print(M_pu,M_pa,offset,drop)
            pass
            raise ValueError
        
        # merge M_pa and M_pu
        M = M_pu + M_pa
        self.part = M
        
        # ....................................................................    

    def __eq__(self, value):
        '''
        Test whether this part and the part 'value' are identical.
        @param
          value: a TetrisPart or its frozen representation.
        '''
        if type(value) is TetrisPart:
            value_array = value.part
        elif type(value) is tuple:
            value_array = np.array(value)
        else:
            return False
        return ( (value_array.shape==self.part.shape)
                and 
                (value_array==self.part).all() )        
            
    def get_frozen(self):
        '''
        Return a representation of this part as a tuple of tuples, 
        by converting the numpy array 'self.part'
        '''
        # use a generator expression
        if self.frozen is None:
            self.frozen = tuple( tuple(r) for r in self.part )
        return self.frozen
                
    def get_height(self):
        '''
        Return the number of rows of this part
        '''
        return self.part.shape[0]
    
    def get_width(self):
        '''
        Return the number of columns of this part
        '''
        return self.part.shape[1]
        
    def display(self, message=None):
        '''
        Display this part with row 0 at the bottom.
        @param
          message : a message to be printed above the part.
        '''
        if message is not None:
            print(message)
        for r in self.part[::-1]:
            for c in r:
                print('{:2} '.format(c),end='')
            else:
                print('')

# ---------------------------------------------------------------------------
    
def make_state_canonical(s):
    '''
    Return the state 's' of a workbench in canonical form.
    
    The parts are sorted according to
        (1) the number of rows
        (2) the number of columns
        (3) the values of their flattened matrices
    
    Although this order is quite arbitrary, it provides a canonical 
    representation of the state of the workbench.

    @param
        s : workbench state, that is a tuple or list of parts
    
    
    @return 
       the state in canonical form (sorted tuple of frozen parts)
        
    '''    
    # L : list of triplets (nrows, ncols, flattened_part)
    L = [ (len(p),len(p[0]), tuple(v for r in p for v in r))  for p in s ]
    L.sort()  # reorder according to the prescribed order
    # T : create from L a tuple of parts representation of the workbench state
    T = [ tuple(
                tuple(t[2][ic+ir*t[1]] for ic in range(t[1]))  
                                            for ir in range(t[0]) 
                ) for t in L ]
    return tuple(T)
           
# ---------------------------------------------------------------------------

def offset_range(pa,pu):
    '''
    Compute the semi-open range [start,end) of legal offset values 
    to build a new part by dropping part 'pa' onto part 'pu'.
    If the offset used is in the interval returned, then the new part will 
    be connected.
    The part 'pa' and 'pu' can be represented either as instances of 
    TetrisPart or as tuples of tuples.
    @param
        pa : part above
        pu : part under
    @return
       start,end
    '''
    # get the widths of 'pa' and 'pu'
    if type(pa) is tuple:
        pa_width = len(pa[0])
    else:
        pa_width = pa.get_width()
    if type(pu) is tuple:
        pu_width = len(pu[0])
    else:
        pu_width = pu.get_width()
    #
    start = -(pa_width-1)
    end = pu_width  # semi-open interval
    return start, end    
    
# ---------------------------------------------------------------------------
    
def display_state(s, message=None):
    '''
    Display this workbench state 
                     (parts with row 0 in contact with the workbench)    
    @param
        s : workbench state
        message : an optional message written before the state
    '''    
    if message is not None:
        print(message)
    # compute the height of the tallest part of 's'
    h_max = max( len(p) for p in s )
    
    for h in range(h_max-1,-1,-1):
        if h==0:
            print('. ',end='')
        else:
            print('  ',end='')
        for p in s:
#            pprint.pprint(p)
            try:
                len_p0 = len( ''.join('{:2} '.format(c)  for c in p[0]) )
            except TypeError:
                print(p)
                print(p[0])    
            if h<len(p):
                # something of the part to show
                for c in p[h]:
                    print('{:2} '.format(c),end='')
            else:
                # print a blank segment
                print(' '*len_p0,end='')
            if h==0:
                print(' . ',end='')
            else:
                print('   ',end='')
        print('') # print new line
                
# ---------------------------------------------------------------------------

def make_random_state(assembly_problem, num_op, display=True):
    '''
    Create a random final state from an initial state by applying
    a number of random (but legal) actions
    @param
        assembly_problem : instance of an AssemblyProblem
        num_op : max number of operations applied
        display : boolean flag. If true, the result is shown at the console.
    @return
            final_state        
    '''      
    ap = assembly_problem
    current_state = ap.initial

    if display:
        print('\n')
        display_state(current_state,"Initial state")
    
    for i in range(num_op):
        la = ap.actions(current_state)
        if len(la)==0:
            break
        ra = random.choice(la)
        current_state = ap.result(current_state, ra)
        if display:
            print('\n')
            display_state(current_state,"After action {} ".format(i+1))
    # 
    return current_state
                                                            
# ---------------------------------------------------------------------------

class AssemblyProblem(generic_search.Problem):
    '''   
    Class that implements the generic interface of the parent class
    'generic_search.Problem'
    
    A state encodes the set of parts present on a workbench as a list of 
    parts in canonical order (see function 'make_state_canonical').
    
    States are tuples as frozenset do not allow repeats.

    An action is a triplet (part_above, part_under, offset).
    If  'part_under' is None then the action 
    encodes  'rotate90(part_above)'
    All the parts are represented as tuples.
    All the states are represented as tuples of parts.
    '''

    def __init__(self, initial, goal=None, use_rotation=False):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments.
        @param
            initial : initial state
            goal : goal state
            use_rotation : boolean flag to indicate whether rotations are 
                           allowed.
        """
        self.use_rotation = use_rotation
        initial = make_state_canonical(initial)
        if goal is not None:
            goal = make_state_canonical(goal)
        # Call the parent class constructor.
        # Here the parent class is 'generic_search.Problem'
        super(AssemblyProblem, self).__init__(initial, goal)

    
    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.        
        """
        #

        raise NotImplementedError


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        # Here a workbench state is a frozenset of parts        
 
        raise NotImplementedError
        
# ---------------------------------------------------------------------------

def play_solution(ap, node):
    '''
    Display all the states visited on the solution path.
    @param
      ap : an instance of an AssemblyProblem
      node : solution node returned by a search algorithm
             of the 'generic_search' module.
    '''
    if node is None:
        print('No solution found!!')
        return
    # La : list of actions
    La = node.solution()  # see class 'Node' in the 'generic_search' module.
    display_state(ap.initial,'\nInitial state\n')
    current_state = ap.initial
    for a in La:
        current_state = ap.result(current_state, a)
        if len(current_state)==1:
            display_state(current_state,'\nGoal state\n')
        else:
            display_state(current_state,' ')
            
# ---------------------------------------------------------------------------
        
def load_state(file_name):
    '''
    Load the workbench state stored in a the text file 'file_name'.
    Each part of the state is represented as a sequence of 
    consecutive lines. Each line entry is space separated.
    For example, a file with the contents
    
            # Comment lines start with the character '#'
            # 
            
            0  2  0
            0  2  0
            2  2  2
            
            0  1  0  1  1
            1  1  3  1  0
            0  3  3  3  0
            0  3  3  0  0
            
    represents a state with two parts.  The function returns the state
    as a tuple of tuples. For instance, with the above contents, the returned
    tuple is
    
        ( 
          ((2, 2, 2), (0, 2, 0), (0, 2, 0)),
          ((0, 3, 3, 0, 0), (0, 3, 3, 3, 0), (1, 1, 3, 1, 0), (0, 1, 0, 1, 1))
        )
    
    WARNINGS: 
        The location of the row in contact with the workbench is different 
        in the text representation and in the tuple representation.
        The file contents is similar to the output of the function
        'display_state'
        
        The state is not necessarily in canonical form.
        Use the function 'make_state_canonical' if needed.
    
    @param
        file_name : the file name of a text file containing a state.
    @return
        a tuple representation of the state contained in the text file.
    '''
    L = []  # list of states
    P = []  # list of rows of the current part
    with open(file_name) as f:
        # process the file line by line
        for line in f:
            # skip comment lines
            if line[0]=='#':
                continue
            # X current row
            X = [int(x) for x in line.split()]
            if len(X)==0:
                # current line is blank
                if len(P)>0:
                    # finalize the current part P
                    L.append(tuple(reversed(P)))
                    P = [] # reset the current part
            else:
                # add the current row to the current part
                P.append(tuple(X))
        # Finalize the current P if needed
        if len(P)>0:
            # finalize the current part P
            L.append(tuple(reversed(P)))
        
    return tuple(L) # return the list of parts as a tuple          
            
# ---------------------------------------------------------------------------
