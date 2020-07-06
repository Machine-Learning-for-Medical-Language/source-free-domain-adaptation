import anafora
import anafora.select


def test_select_all():
    select = anafora.select.Select()

    assert select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert select('C')


def test_select_type():
    select = anafora.select.Select(include={'A', 'B'})

    assert select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(exclude=[('A',), ('B',)])

    assert not select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(include={'A'}, exclude={'B'})

    assert select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')


def test_select_prop_name():
    select = anafora.select.Select(include=[('A',), ('B', 'R')])

    assert select('A')
    assert not select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(exclude={'A', ('B', 'R')})

    assert not select('A')
    assert select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(include={'B'}, exclude={('B', 'P')})

    assert not select('A')
    assert not select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')


def select_prop_value():
    select = anafora.select.Select(include={'A', ('B', 'R', 'V')})

    assert select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(exclude={'A', ('B', 'R', 'V')})

    assert not select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert not select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(include={'C', ('C', 'R', 'W')})

    assert not select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert select('C')


def test_select_star():
    select = anafora.select.Select(include={'*'})

    assert select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(exclude={'*'})

    assert not select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(include={('*', 'R')})

    assert not select('A')
    assert not select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(include=[('C',)], exclude=[('C', '*')])

    assert not select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert not select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(include=['C', 'B'], exclude=[('C', '*'), ('B', '*', '*')])

    assert not select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert not select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(exclude={('C', 'R', '*')})

    assert select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert select('C')

    select = anafora.select.Select(include={('*', '*', '*')})

    assert not select('A')
    assert not select('B', 'P')
    assert not select('B', 'R')
    assert select('B', 'R', 'V')
    assert select('C', 'R', 'V')
    assert select('C', 'R', 'W')
    assert not select('C')

    select = anafora.select.Select(exclude={('*', '*', '*')})

    assert select('A')
    assert select('B', 'P')
    assert select('B', 'R')
    assert not select('B', 'R', 'V')
    assert not select('C', 'R', 'V')
    assert not select('C', 'R', 'W')
    assert select('C')
