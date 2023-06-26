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
        ('Tot_Rept_Positions_Long_All',7),
        ('Tot_Rept_Positions_Short_All',8),
        ('M_Money_Positions_Long_All',9),
        ('M_Money_Positions_Short_All',10),
    )
    
class PandasData_more(btfeeds.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''
    lines = ('Tot_Rept_Positions_Long_All', 'Tot_Rept_Positions_Short_All',
             'M_Money_Positions_Long_All','M_Money_Positions_Short_All',)

    params = (
        ('Tot_Rept_Positions_Long_All',-1),
        ('Tot_Rept_Positions_Short_All',-1),
        ('M_Money_Positions_Long_All',-1),
        ('M_Money_Positions_Short_All',-1),
    )