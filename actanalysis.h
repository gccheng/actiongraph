#ifndef ACTANALYSIS_H
#define ACTANALYSIS_H

#include <QMainWindow>

namespace Ui {
class ActAnalysis;
}

class ActAnalysis : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit ActAnalysis(QWidget *parent = 0);
    ~ActAnalysis();
    
private:
    Ui::ActAnalysis *ui;
};

#endif // ACTANALYSIS_H
