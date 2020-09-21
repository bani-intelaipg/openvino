# Bani
set(TARGET_NAME "tfbridge_ngfunc")

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ie_add_sample(NAME tfbridge_ngfunc
              SOURCES ${MAIN_SRC}
              HEADERS ${MAIN_HEADERS}
              DEPENDENCIES format_reader)

find_package(ngraph REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES})