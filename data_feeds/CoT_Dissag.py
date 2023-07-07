import backtrader.feeds as btfeeds

class OHLCAVWithDisaggCoT(btfeeds.GenericCSVData):
  
    lines = ('Tot_Rept_Positions_Long_All',
            'M_Money_Positions_Long_All',
            'Tot_Rept_Positions_Short_All',
            'M_Money_Positions_Short_All',)
    

    params = (
        ('nullvalue', 0.0),
        ('dtformat', ('%Y-%m-%d')),

        ('Date', None),
        ('Open', 1),
        ('High', 2),
        ('Low', 3),
        ('Close', 4),
        ('Adj Close', 5),
        ('Volume', 6),
        ('Col_1',7),
        ('Col_2',8),
        ('Col_3',9),
        ('Col_4',10),
        ('Col_5',11),
        ('Col_6',12),
        ('Col_7',13),
    )
    
class PandasData_more(btfeeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''
    lines = ('Tot_Rept_Positions_Long_All', 'Tot_Rept_Positions_Short_All',
             'M_Money_Positions_Long_All','M_Money_Positions_Short_All',
            'Pct_of_OI_M_Money_Short_All','Pct_of_OI_M_Money_Long_All',
            'Pct_of_OI_Tot_Rept_Short_All','Pct_of_OI_Tot_Rept_Long_All',)

    params = (
        ('Tot_Rept_Positions_Long_All',-1),
        ('Tot_Rept_Positions_Short_All',-1),
        ('M_Money_Positions_Long_All',-1),
        ('M_Money_Positions_Short_All',-1),
        ('Pct_of_OI_M_Money_Short_All',-1),
        ('Pct_of_OI_M_Money_Long_All',-1),
        ('Pct_of_OI_Tot_Rept_Short_All',-1),
        ('Pct_of_OI_Tot_Rept_Long_All',-1),
    )


class DataFeedForAI(btfeeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''
    lines = ('Col_1', 'Col_2',
             'Col_3','Col_4',
            'Col_5','Col_6',
            'Col_7',)

    params = (
        ('Col_1',-1),
        ('Col_2',-1),
        ('Col_3',-1),
        ('Col_4',-1),
        ('Col_5',-1),
        ('Col_6',-1),
        ('Col_7',-1),
    )