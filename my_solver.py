# -*- coding: utf-8 -*-

import numpy as np

import itertools

import generic_search

from Helper_functions import (TetrisPart, AssemblyProblem, offset_range, 
                            display_state, 
                            make_state_canonical, play_solution, 
                            load_state, make_random_state
                            )
    
# ---------------------------------------------------------------------------
def appear_as_subpart(some_part, goal_part):
    '''    
    Determine whether the part 'some_part' appears in another part 'goal_part'.
    
    Formally, we say that 'some_part' appears in another part 'goal_part',
    when the matrix representation 'S' of 'some_part' is a a submatrix 'M' of
    the matrix representation 'G' of 'goal_part' and the following constraints
    are satisfied:
        for all indices i,j
            S[i,j] == 0 or S[i,j] == M[i,j]
            
    During an assembly sequence that does not use rotations, any part present 
    on the workbench has to appear somewhere in a goal part!
    
    @param
        some_part: a tuple representation of a tetris part
        goal_part: a tuple representation of another tetris part
        
    @return
        True if 'some_part' appears in 'goal_part'
        False otherwise    
    '''
    #taking some_parts to intial array
    comparing=np.array(some_part)
    #taking goal_parts to goal array
    goal=np.array(goal_part)
    
    #taking row and column sizes of gaol and initial parts
     
    try:
        comparing_row=comparing.shape[0]   #assigning the number of rows in initial part to initial_row variable
    except IndexError:
        comparing_row=0                 #if an error occours 0 will be assigned to the variable
          
    try:
        comparing_col=comparing.shape[1]   #assigning the number of columns in initial part to initial_col variable
    except IndexError:
        comparing_col=0                 #if an error occours 0 will be assigned to the variable   
        
    try:
        goal_row=goal.shape[0]       #assigning the number of rowss in goal part to goal_row variable
    except IndexError:
        goal_row=0                   #if an error occours 0 will be assigned to the variable 
        
    try:
        goal_col=goal.shape[1]       #assigning the number of columns in goal part to goal_col variable
    except IndexError:
        goal_col=0                   #if an error occours 0 will be assigned to the variable 
        
        
#    print('comparing_row =',comparing_row,'comparing_col =',comparing_col,'goal_row =',goal_row,'goal_col =',goal_col)  #debug message
    
    #check the simialarity between parts only if goal and initial part have values
    if(goal_row!=0 and goal_col!=0 and (comparing_row!=0 or comparing_col!=0)):
        row_range =goal_row-comparing_row+1        #difference between the ro sizes of goal and initail part
        col_range =goal_col-comparing_col+1        #difference between the ro sizes of goal and initail part
        
        for i,j in itertools.product(range(0,row_range),range(0,col_range)):   #nested for loop to loop through every 0 to the row range and column range
#            print('i =',i,',j =',j)
            sub_part=goal[i:(i+comparing_row),j:(j+comparing_col)]       #extracting a subpart of a size of comparing part from the goal        
            similar=((comparing==0)|(sub_part==comparing))               #checking if each value of the comparing part is equal to subpart taken from the goal
            
            if (similar.all() ==True):          #if all the values checked are true function will return True
                return True 
#        print('num is =',num)
    return False                                #if comparing part doesnt exist in the goal part function returns False
        
def cost_rotated_subpart(some_part, goal_part):
    '''    
    Determine whether the part 'some_part' appears in another part 'goal_part'
    as a rotated subpart. If yes, return the number of 'rotate90' needed, if 
    no return 'np.inf'
    
    The definition of appearance is the same as in the function 
    'appear_as_subpart'.
                   
    @param
        some_part: a tuple representation of a tetris part
        goal_part: a tuple representation of another tetris part
    
    @return
        the number of rotation needed to see 'some_part' appear in 'goal_part'
        np.inf  if no rotated version of 'some_part' appear in 'goal_part'
    
    '''
    #if the part appear in the goal without any rotation function returns 0
    if appear_as_subpart(some_part, goal_part) :
        return 0
    
    #in a for loop rotatong the some_part 3 times and check if a rotated version of some_part appears in the goal
    for i in range (1,4):
        rotating_tetris=TetrisPart(some_part)      #making a tetris part from the some_part
        rotating_tetris.rotate90()                 #rotating the tetris by 90 gedrees
#        rotating_tetris.display('dd')
        some_part=rotating_tetris.get_frozen()          #converting tetris into tuple form by calling get_frozen function
        
         #if the rotated version of some_part appear in the goal function return the number of rotations to achieve similarity
        if appear_as_subpart(some_part, goal_part) :   
            return i
    #if the rotated version of some_part doesn't appear in the goal function return np.inf
    return np.inf
    
# ---------------------------------------------------------------------------

class AssemblyProblem_1(AssemblyProblem):
    '''
    
    Subclass of 'assignment_one.AssemblyProblem'
    
    * The part rotation action is not available for AssemblyProblem_1 *

    The 'actions' method of this class simply generates
    the list of all legal actions. The 'actions' method of this class does 
    *NOT* filtered out actions that are doomed to fail. In other words, 
    no pruning is done in the 'actions' method of this class.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_1, self).__init__(initial, goal, use_rotation=False)
        
      
            
    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        @param
          state : a state of an assembly problem.
        
        @return 
           the list of all legal drop actions available in the 
            state passed as argument.
        
        """

        action_list = []                                                        #creating a list to store actions
        state_list=list(state)                                                  #taking the states into a list
        list_length=len(state_list)                                             #taking the length of the state list
        
        #in a nested for loop considering all the combinations of two objects with different offsets and storing them in action_list
        #itertools.permutation is used here because it doesn't run the for loop when the i and j values are equal
        #in other words this loop would not try to combine the two copies of the same part 
        for i,j in itertools.permutations(range(0,list_length),2):     
            
            part_above = state_list[i]                                          #taking the i th item in the state_list
            part_under = state_list[j]                                          #taking the j th item in the state_list
            range_start, range_end = offset_range(part_above,part_under)        #minimum and maximum values for legal offset values
            
            for offset in range(range_start, range_end):                        #in a for loop considering all legal offsets 
                
                #testing if in the new part that can be make in this combination part above and part below is touching each other
                if offset != None:                                         
                    action_list.append((part_above,part_under, offset))         #if it touch each other add the combination in to the action list
                    
        #returning the action list
        return action_list


    def result(self, state, action):
        """
        Return the state (as a tuple of parts in canonical order)
        that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        
        @return
          a state in canonical order
        
        """
        
        state_list=list(state)                          #taking states into states_list
        part_above,part_under, offset = action                         #taking sets of part_above , part_under and offset values into seperate lists from the action list
#        print('assembly 1 pa =',pa)
#        print('assembly 1 pu =',pu)
#        print('assembly 1 offset =',offset)
        new_Tetris = TetrisPart(part_above,part_under,offset)           #combining sets of part abave,part under and offset to create new parts
        new_Tetris_part = new_Tetris.get_frozen()       #calling get frozen on the new part    
        state_list.remove(part_above)                           #removing items of part above list from the states list
        state_list.remove(part_under)                           #removing items of part below list from the states list
        state_list.append(new_Tetris_part)              #adding new tetris part into the list
        state_tuple = tuple(state_list)                 #conveting the state list into a tuple called state tuple
        state_tuple = make_state_canonical(state_tuple) #sorting that state tuple into conical order
        return state_tuple                              #returning state_tuple

# ---------------------------------------------------------------------------

class AssemblyProblem_2(AssemblyProblem_1):
    '''
    
    Subclass of 'assignment_one.AssemblyProblem'
        
    * Like for AssemblyProblem_1,  the part rotation action is not available 
       for AssemblyProblem_2 *

    The 'actions' method of this class  generates a list of legal actions. 
    But pruning is performed by detecting some doomed actions and 
    filtering them out.  That is, some actions that are doomed to 
    fail are not returned. In this class, pruning is performed while 
    generating the legal actions.
    However, if an action 'a' is not doomed to fail, it has to be returned. 
    In other words, if there exists a sequence of actions solution starting 
    with 'a', then 'a' has to be returned.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_2, self).__init__(initial, goal)
    
    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        A candidate action is eliminated if and only if the new part 
        it creates does not appear in the goal state.
        """
        action_list = []                                                        #creating a list to store actions
        state_list=list(state)                                                  #taking the states into a list
        list_length=len(state_list)                                             #taking the length of the state list

        #in a nested for loop considering all the combinations of two objects with different offsets and storing them in action_list
        #itertools.permutation is used here because it doesn't run the for loop when the i and j values are equal
        #in other words this loop would not try to combine the two copies of the same part
        for i,j in itertools.permutations(range(0,list_length),2):
            part_above = state_list[i]                                          #taking the i th item in the state_list
            part_under = state_list[j]                                          #taking the j th item in the state_list
            range_start, range_end = offset_range(part_above,part_under)        #minimum and maximum values for legal offset values
            
            for offset in range(range_start, range_end):                        #in a for loop considering all legal offsets 
                
                #testing if in the new part that can be make in this combination part above and part below is touching each other
                if offset != None:
                    
                    New_Tetris = TetrisPart(part_above,part_under,offset)       #combining sets of part abave,part under and offset to create new parts
                    New_Tetris_tuple=New_Tetris.get_frozen()                    #calling get frozen on the new part 
                    
                    #checking if the newly created part appear as a subpart in the goal part
                    if appear_as_subpart(New_Tetris_tuple, self.goal[0]):        
                        action_list.append((part_above,part_under, offset))     #if so add the combination to the action list   
                        
        #returning the action list
        return action_list
    
# ---------------------------------------------------------------------------

class AssemblyProblem_3(AssemblyProblem_1):
    '''
    
    Subclass 'assignment_one.AssemblyProblem'
    
    * The part rotation action is available for AssemblyProblem_3 *

    The 'actions' method of this class simply generates
    the list of all legal actions including rotation. 
    The 'actions' method of this class does 
    *NOT* filter out actions that are doomed to fail. In other words, 
    no pruning is done in this method.
        
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_3, self).__init__(initial, goal)
        self.use_rotation = True

    
    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        Rotations are allowed, but no filtering out the actions that 
        lead to doomed states.
        
        """

        action_list = []                                                        #creating a list to store actions
        state_list=list(state)                                                  #taking the states into a list
        list_length=len(state_list)                                             #taking the length of the state list
        
        previous_pa=None                                                        #initiating previous_pa as none
        
        #in a nested for loop considering all the combinations of two objects with different offsets and storing them in action_list
        #itertools.permutation is used here because it doesn't run the for loop when the i and j values are equal
        #in other words this loop would not try to combine the two copies of the same part
        for i,j in itertools.permutations(range(0,list_length),2):
            part_above = state_list[i]                                          #taking the i th item in the state_list
            part_under = state_list[j]                                          #taking the j th item in the state_list
            
            range_start, range_end = offset_range(part_above,part_under)        #minimum and maximum values for legal offset values
            
            
            for offset in range(range_start, range_end):                        #in a for loop considering all legal offsets 
                
                #testing if in the new part that can be make in this combination part above and part below is touching each other
                if offset != None: 
                    action_list.append((part_above,part_under, offset))         #if it touches each other, then add the combination in to the action list
            if not part_above==previous_pa:                                     #if the pa is not the same one as the last test
                action_list.append((part_above, None, 0))                    #add only part abave with offset 0 to check if a rotated version of it appears in the goal part
            previous_pa=part_above                                              #assign current part above to previous_pa variable
        #return action list
        return action_list
#

        
    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        The action can be a drop or rotation.        
        """

        state_list=list(state)                              #taking states into states_list

        part_above, part_under, offset = action                             #taking sets of part_above , part_under and offset values into seperate lists from the action list
        if( part_under == None ):                                   #if value for pu doesnt exist
            new_Tetris = TetrisPart(part_above)                     #create new parts with only part_above without a offset
            new_Tetris.rotate90()                           #rotate new tetris by 90 degrees
            new_Tetris_part=new_Tetris.get_frozen()         #calling get frozen on the new part
            state_list.append(new_Tetris_part)              #adding new tetris part into the list
            state_list.remove(part_above)                           #removing items of part above list from the states list
        else:
            new_Tetris = TetrisPart(part_above,part_under,offset)           #combining sets of part abave,part under and offset to create new parts
            new_Tetris_part = new_Tetris.get_frozen()       #calling get frozen on the new part
            state_list.append(new_Tetris_part)              #adding new tetris part into the list        
            state_list.remove(part_above)                           #removing items of part above list from the states list
            state_list.remove(part_under)                           #removing items of part under list from the states list
        
        state_tuple = tuple(state_list)                     #conveting the state list into a tuple called state tuple
        state_tuple = make_state_canonical(state_tuple)     #sorting that state tuple into conical order
        return state_tuple                                  #returning state_tuple

# ---------------------------------------------------------------------------

class AssemblyProblem_4(AssemblyProblem_3):
    '''
    
    Subclass 'assignment_one.AssemblyProblem3'
    
    * Like for its parent class AssemblyProblem_3, 
      the part rotation action is available for AssemblyProblem_4  *

    AssemblyProblem_4 introduces a simple heuristic function and uses
    action filtering.
    See the details in the methods 'self.actions()' and 'self.h()'.
    
    '''

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        # Call the parent class constructor.
        # Here the parent class is 'AssemblyProblem' 
        # which itself is derived from 'generic_search.Problem'
        super(AssemblyProblem_4, self).__init__(initial, goal)

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        
        Filter out actions (drops and rotations) that are doomed to fail 
        using the function 'cost_rotated_subpart'.
        A candidate action is eliminated if and only if the new part 
        it creates does not appear in the goal state.
        This should  be checked with the function "cost_rotated_subpart()'.
                
        """        

        action_list = []                                                        #creating a list to store actions
        state_list=list(state)                                                  #taking the states into a list
        list_length=len(state_list)                                             #taking the length of the state list
        
        previous_pa=None                                                       #initiating previous_pa as none
        
        #in a nested for loop considering all the combinations of two objects with different offsets and storing them in action_list
        #itertools.permutation is used here because it doesn't run the for loop when the i and j values are equal
        #in other words this loop would not try to combine the two copies of the same part
        for i,j in itertools.permutations(range(0,list_length),2):
            
            part_above = state_list[i]                                          #taking the i th item in the state_list
            part_under = state_list[j]                                          #taking the j th item in the state_list
            range_start, range_end = offset_range(part_above,part_under)        #minimum and maximum values for legal offset values
            
            for offset in range(range_start, range_end):                        #in a for loop considering all legal offsets 
                
                #testing if in the new part that can be make in this combination part above and part below is touching each other
                if offset!= None:
                    
                    New_Tetris = TetrisPart(part_above,part_under,offset)       #combining sets of part abave,part under and offset to create new parts
                    New_Tetris_tuple=New_Tetris.get_frozen()                    #calling get frozen on the new part 
                    
                    #checking if the newly created part appear as a subpart in the goal part
                    if not cost_rotated_subpart(New_Tetris_tuple, self.goal[0])== np.inf:
                        
                        action_list.append((part_above,part_under, offset))     #if so add the combination to the action list 
            if not part_above==previous_pa:                                     #if the pa is not the same one as the last test
                if not cost_rotated_subpart(part_above, self.goal[0])== np.inf:
                    action_list.append((part_above, None, 0))                    #add only part abave with offset 0 to check if a rotated version of it appears in the goal part
            previous_pa=part_above                                              #assign current part above to previous_pa variable
            
        #returning the action list    
        return action_list
    
        
    def h(self, n):
        '''
        This heuristic computes the following cost; 
        
           Let 'k_n' be the number of parts of the state associated to node 'n'
           and 'k_g' be the number of parts of the goal state.
          
        The cost function h(n) must return 
            k_n - k_g + max ("cost of the rotations")  
        where the list of cost of the rotations is computed over the parts in 
        the state 'n.state' according to 'cost_rotated_subpart'.
        
        
        @param
          n : node of a search tree
          
        '''

        k_n=len(n.state)                                            #getting the number of parts of the state associated to node 'n'
        k_g=len(self.goal)                                          #getting the number of parts of the goal state
        node_list = n.state                                         #taking the parts of the state associated to node 'n' into a list
        #calculating the cost of rotations for each part
        Rotates_cost = [cost_rotated_subpart(comparing_part, self.goal[0]) for comparing_part in node_list]  
        
        max_cost=max(Rotates_cost)                                  #getting the maximum rotation cost from the Rotate_costs list
        h_value= k_n - k_g + max_cost                               #calculating the heuristic
        return h_value                                              #calculating heuristics value

# ---------------------------------------------------------------------------
        
def solve_1(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_1
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_1
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''
    print('\n++  busy searching in solve_1() ...  ++\n')

    
    assembly_problem = AssemblyProblem_1(initial, goal)                             #initiating assembly problem 1 with initial workbench and goal
    search_function = generic_search.breadth_first_tree_search(assembly_problem)    #using breadth first tree search to solve the find a solution
    search_function1 = generic_search.depth_first_tree_search(assembly_problem)
    search_function2 = generic_search.breadth_first_graph_search(assembly_problem)
    search_function3 = generic_search.uniform_cost_search(assembly_problem)
    search_function4 = generic_search.iterative_deepening_search(assembly_problem)
    
    if search_function == None:                                     #if search does not output anything string 'no solution' will be returned from this function
        return 'no solution'
    else:
        return search_function.solution()#, search_function1.solution(), search_function2.solution(), search_function3.solution(), search_function4.solution()                          #otherwise the solution will be returned 
            

# ---------------------------------------------------------------------------
        
def solve_2(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_2
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_2
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''

    print('\n++  busy searching in solve_2() ...  ++\n')
    
    assembly_problem=AssemblyProblem_2(initial, goal)                               #initiating assembly problem 1 with initial workbench and goal
    
    search_function = generic_search.breadth_first_tree_search(assembly_problem)   #using breadth first tree search to solve the find a solution
    search_function1 = generic_search.depth_first_tree_search(assembly_problem)
    search_function2 = generic_search.breadth_first_graph_search(assembly_problem)
    search_function3 = generic_search.uniform_cost_search(assembly_problem)
    search_function4 = generic_search.iterative_deepening_search(assembly_problem)
    
    if search_function == None:                                     #if search does not output anything string 'no solution' will be returned from this function
        return 'no solution'
    else:
        return search_function.solution()#, search_function1.solution(), search_function2.solution(), search_function3.solution(), search_function4.solution()                           #otherwise the solution will be returned 

# ---------------------------------------------------------------------------
        
def solve_3(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_3
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_3
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''

    print('\n++  busy searching in solve_3() ...  ++\n')
    
    assembly_problem=AssemblyProblem_3(initial, goal)                             #initiating assembly problem 1 with initial workbench and goal
    
    search_function = generic_search.breadth_first_tree_search(assembly_problem)    #using breadth first graph search to solve the find a solution
    search_function1 = generic_search.depth_first_tree_search(assembly_problem)
    search_function2 = generic_search.breadth_first_graph_search(assembly_problem)
    search_function3 = generic_search.uniform_cost_search(assembly_problem)
    search_function4 = generic_search.iterative_deepening_search(assembly_problem)
    
    if search_function == None:                                     #if search does not output anything string 'no solution' will be returned from this function
        return 'no solution'
    else:
        return search_function.solution()#, search_function1.solution(), search_function2.solution(), search_function3.solution(), search_function4.solution()                           #otherwise the solution will be returned 
    
# ---------------------------------------------------------------------------
        
def solve_4(initial, goal):
    '''
    Solve a problem of type AssemblyProblem_4
    
    The implementation has to 
    - use an instance of the class AssemblyProblem_4
    - make a call to an appropriate functions of the 'generic_search" library
    
    @return
        - the string 'no solution' if the problem is not solvable
        - otherwise return the sequence of actions to go from state
        'initial' to state 'goal'
    
    '''
    print('\n++  busy searching in solve_4() ...  ++\n')
    assembly_problem = AssemblyProblem_4(initial, goal)                                         #initiating assembly problem 1 with initial workbench and goal
    
    search_function = generic_search.astar_graph_search(assembly_problem,assembly_problem.h)    #using A star search to solve the find a solution
    search_function2 = generic_search.breadth_first_graph_search(assembly_problem)
    search_function3 = generic_search.uniform_cost_search(assembly_problem)
    search_function4 = generic_search.iterative_deepening_search(assembly_problem)
    
    if search_function == None:                                     #if search does not output anything string 'no solution' will be returned from this function
        return 'no solution'
    else:
        return search_function.solution()#,search_function2.solution(), search_function3.solution(), search_function4.solution()                           #otherwise the solution will be returned 
        
# ---------------------------------------------------------------------------


    
if __name__ == '__main__':
    pass
    
