#-------------------------------------------------
#
# Project created by QtCreator 2013-04-04T16:28:25
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = ActivityAnalysis
TEMPLATE = app

#QMAKE CXXFLAGS += -std=c++0x
#CXXFLAGS="-std=c++0x"
CXXFLAGS += -std=c++0x

SOURCES += main.cpp\
        actanalysis.cpp \
    densetrack.cpp \
    imagepyramid.cpp \
    actgraph.cpp \
    utility.cpp

HEADERS  += actanalysis.h \
    densetrack.h \
    track.h \
    initialize.h \
    imagepyramid.h \
    imagepyramid.hpp \
    actgraph.h \
    utility.h

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_ml -lopencv_videoio -lopencv_video -lopencv_objdetect


FORMS    += actanalysis.ui

OTHER_FILES += \
    notes.txt
