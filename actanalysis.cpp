#include "actanalysis.h"
#include "ui_actanalysis.h"

ActAnalysis::ActAnalysis(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ActAnalysis)
{
    ui->setupUi(this);
}

ActAnalysis::~ActAnalysis()
{
    delete ui;
}
