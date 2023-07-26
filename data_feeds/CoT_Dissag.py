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
        ('Predict_1',7),
        ('Predict_2',8),
        ('Predict_3',9),
        ('Predict_4',10),
        ('Predict_5',11),
        ('Predict_6',12),
        ('Predict_7',13),
        ('signal',14),
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
    lines = ('Predict_1', 'Predict_2',
             'Predict_3','Predict_4',
            'Predict_5','Predict_6',
            'Predict_7', 'signal')

    params = (
        ('Predict_1',-1),
        ('Predict_2',-1),
        ('Predict_3',-1),
        ('Predict_4',-1),
        ('Predict_5',-1),
        ('Predict_6',-1),
        ('Predict_7',-1),
        ('signal',-1),
    )