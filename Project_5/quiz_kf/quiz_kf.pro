TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        kalman_filter.cpp \
        main.cpp \
        tracking.cpp

HEADERS += \
  kalman_filter.h \
  measurement_package.h \
  tracking.h
