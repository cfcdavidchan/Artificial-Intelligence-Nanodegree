assignments = []


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def naked_twins(values):
    double_digit = []
    double_digit_box = []
    for box, value in values.items():
        if len(value) == 2:
            double_digit.append(value)
            double_digit_box.append(box)

    i = 0
    double_digit_dict = dict()
    while i < len(double_digit_box):
        if double_digit[i] not in double_digit_dict.keys():
            double_digit_dict[double_digit[i]] = [double_digit_box[i]]
        else:
            double_digit_dict[double_digit[i]].append(double_digit_box[i])
        i += 1

    for digit, all_box in double_digit_dict.items():
        for box in all_box:
            row = row_unit_peer(box,row_units)
            column = cols_unit_peer(box,column_units)
            square = square_unit_peer(box,square_units)

            row = [boxes for boxes in row if box != boxes]
            column = [boxes for boxes in column if box != boxes]
            square = [boxes for boxes in square if box != boxes]

            row_twins =[boxes for boxes in row if values[boxes] == digit]
            if len(row_twins):
                for number in digit:
                    for boxs in row:
                        if boxs != row_twins[0]:
                            values[boxs] = values[boxs].replace(number, '')

            column_twins = [boxes for boxes in column if values[boxes] == digit]
            if len(column_twins):
                for number in digit:
                    for boxs in column:
                        if boxs != column_twins[0]:
                            values[boxs] = values[boxs].replace(number, '')

            square_twins = [boxes for boxes in square if values[boxes] == digit]
            if len(square_twins):
                for number in digit:
                    for boxs in square:
                        if boxs != square_twins[0]:
                            values[boxs] = values[boxs].replace(number, '')

    return values
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s + t for s in A for t in B]

rows = 'ABCDEFGHI'
cols = '123456789'
boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
diagonal_units = [[r + str(rows.index(r) + 1) for r in rows], [r + str(9 - rows.index(r)) for r in rows]]

unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)

def row_unit_peer(box,row_units):
    for row in row_units:
        if box in row:
            return row

def cols_unit_peer(box,column_units):
    for column in column_units:
        if box in column:
            return column

def square_unit_peer(box,square_units):
    for square in square_units:
        if box in square:
            return square

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    print


def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            #             values[peer] = values[peer].replace(digit,'')

            values = assign_value(values, peer, values[peer].replace(digit, ''))

    return values


def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                #                 values[dplaces[0]] = digit
                values = assign_value(values, dplaces[0], digit)
    return values


def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values  ## Solved!
    # Chose one of the unfilled square s with the fewest possibilities
    n, s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return (attempt)


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """



    values = grid_values(grid)
    return search(values)


if __name__ == '__main__':


    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments

        visualize_assignments(assignments)

    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')