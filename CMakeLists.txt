cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_demo)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

set(rknpu_yolov8_file rknpu2/yolov8.cc)

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
    set(rknpu_yolov8_file rknpu2/yolov8_rv1106_1103.cc)
    #dma
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma)
endif()

if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    add_definitions(-DRKNPU1)
    set(rknpu_yolov8_file rknpu1/yolov8.cc)
endif()

# 添加3rdparty目录（如果存在）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
endif()

# 添加utils目录（如果存在）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/utils/CMakeLists.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# 设置RKNN库路径
set(LIBRKNNRT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include)
set(LIBRKNNRT ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# 添加单图片检测版本（main.cc）
add_executable(${PROJECT_NAME}
    main.cc
    src/postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    ${rknpu_yolov8_file}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
)

# 只保留多线程视频检测版本，移除单线程图片检测和视频检测


# 移除zero_copy版本，只保留多线程版本

# 安全帽检测
add_executable(helmet_detection
    src/helmet_detection.cc
    src/postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    src/camera_resource_pool.cc
    ${rknpu_yolov8_file}
)

# 链接安全帽检测程序
target_link_libraries(helmet_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    pthread
    rt
)

target_include_directories(helmet_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(helmet_detection
    log
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(helmet_detection Threads::Threads)
endif()

# 火焰检测
add_executable(flame_detection
    src/flame_detection.cc
    src/postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    src/camera_resource_pool.cc
    ${rknpu_yolov8_file}
)

# 吸烟检测
add_executable(smoking_detection
    src/smoking_detection.cc
    src/postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    src/camera_resource_pool.cc
    ${rknpu_yolov8_file}
)

# 人脸检测
add_executable(face_detection
    src/face_detection.cc
    src/postprocess.cc
    src/retinaface_postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    ${rknpu_yolov8_file}
)

# 多任务检测（单窗口）
add_executable(multi_task_detection
    src/multi_task_detection.cc
    src/postprocess.cc
    src/retinaface_postprocess.cc
    src/meter_postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    ${rknpu_yolov8_file}
)

# 多窗口检测系统
add_executable(multi_window_detection
    src/multi_window_detection.cc
    src/postprocess.cc
    src/retinaface_postprocess.cc
    src/meter_postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    ${rknpu_yolov8_file}
)

# 生产者-消费者检测系统
add_executable(producer_consumer_detection
    src/main.cpp
    src/camera.cpp
    src/postprocess.cc
    src/retinaface_postprocess.cc
    src/meter_postprocess.cc
    src/image_utils.cc
    src/file_utils.cc
    src/image_drawing.cc
    src/shared_camera_manager.cc
    src/camera_detector.cc
    ${rknpu_yolov8_file}
)

target_include_directories(producer_consumer_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_link_libraries(producer_consumer_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(multi_task_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_link_libraries(multi_task_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    rt
    pthread
)

target_include_directories(multi_window_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_link_libraries(multi_window_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    rt
)




target_include_directories(flame_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_include_directories(smoking_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)

target_include_directories(face_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBRKNNRT_INCLUDES}
)


target_link_libraries(flame_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    rt
)

target_link_libraries(smoking_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    rt
)

target_link_libraries(face_detection
    ${OpenCV_LIBS}
    ${LIBRKNNRT}
    dl
    rt
)


if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(flame_detection
    log
    )
    target_link_libraries(smoking_detection
    log
    )
    target_link_libraries(face_detection
    log
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(helmet_detection Threads::Threads)
    target_link_libraries(flame_detection Threads::Threads)
    target_link_libraries(smoking_detection Threads::Threads)
    target_link_libraries(face_detection Threads::Threads)
    target_link_libraries(multi_task_detection Threads::Threads)
    target_link_libraries(multi_window_detection Threads::Threads)
    target_link_libraries(producer_consumer_detection Threads::Threads)
endif()

install(TARGETS ${PROJECT_NAME} helmet_detection flame_detection smoking_detection face_detection multi_task_detection multi_window_detection producer_consumer_detection DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/helmet_labels_list.txt DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/flame_labels_list.txt DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/smoking_labels_list.txt DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/face_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)