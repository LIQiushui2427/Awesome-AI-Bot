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
        ('Pct_of_OI_M_Money_Short_All',11),
        ('Pct_of_OI_M_Money_Long_All',12),
        ('Pct_of_OI_Tot_Rept_Short_All',13),
        ('Pct_of_OI_Tot_Rept_Long_All',14),
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