"""
debug realsense d435i
generated with Claude Code
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def show_stereo_cameras():
    """D435Iの左右カメラを表示する基本版"""
    print("D435I 左右カメラを起動します...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 左右の赤外線カメラを有効化
    # index=1: 左カメラ, index=2: 右カメラ
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 左
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 右
    
    try:
        pipeline.start(config)
        print("✓ ステレオカメラストリーミング開始")
        
        # ウォームアップ
        for _ in range(10):
            pipeline.wait_for_frames()
        print("✓ ウォームアップ完了")
        
        print("左右カメラ表示中... 'q'で終了")
        
        while True:
            frames = pipeline.wait_for_frames()
            
            # 左右の赤外線フレームを取得
            left_frame = frames.get_infrared_frame(1)   # 左カメラ
            right_frame = frames.get_infrared_frame(2)  # 右カメラ
            
            if not left_frame or not right_frame:
                continue
            
            # numpy配列に変換
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())
            
            # 左右の画像を横に並べる
            stereo_image = np.hstack((left_image, right_image))
            
            # ラベルを追加
            cv2.putText(stereo_image, 'Left Camera', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(stereo_image, 'Right Camera', (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            cv2.imshow('D435I Stereo Cameras', stereo_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        print("ストリーミング終了")

def show_all_streams():
    """カラー + 左右カメラ + デプスを全部表示"""
    print("全ストリーム表示モード...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 全ストリームを有効化
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)     # カラー
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      # デプス
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30) # 左
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30) # 右
    
    try:
        pipeline.start(config)
        print("✓ 全ストリーミング開始")
        
        # ウォームアップ
        for _ in range(20):
            pipeline.wait_for_frames()
        
        while True:
            frames = pipeline.wait_for_frames()
            
            # 全フレームを取得
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if not all([color_frame, depth_frame, left_frame, right_frame]):
                continue
            
            # numpy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())
            
            # デプスをカラーマップに変換
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 左右の赤外線画像をカラーに変換（表示用）
            left_color = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
            right_color = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
            
            # 2x2のグリッドに配置
            top_row = np.hstack((color_image, depth_colormap))
            bottom_row = np.hstack((left_color, right_color))
            full_image = np.vstack((top_row, bottom_row))
            
            # ラベルを追加
            cv2.putText(full_image, 'Color', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(full_image, 'Depth', (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(full_image, 'Left IR', (10, 510), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(full_image, 'Right IR', (650, 510), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('D435I All Streams', full_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

def stereo_depth_analysis():
    """左右カメラの視差を使ったデプス解析"""
    print("ステレオ視差解析モード...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
        print("✓ ステレオ解析モード開始")
        
        # OpenCVのステレオマッチング
        stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        for _ in range(20):
            pipeline.wait_for_frames()
        
        while True:
            frames = pipeline.wait_for_frames()
            
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            depth_frame = frames.get_depth_frame()
            
            if not all([left_frame, right_frame, depth_frame]):
                continue
            
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # OpenCVでステレオマッチング
            disparity = stereo.compute(left_image, right_image)
            
            # 視差マップを正規化
            disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
            
            # RealSenseのデプスと比較
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 結果を並べて表示
            comparison = np.hstack((disparity_color, depth_colormap))
            
            cv2.putText(comparison, 'OpenCV Stereo', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, 'RealSense Depth', (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Stereo Comparison', comparison)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"エラー: {e}")
        
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

def high_fps_stereo():
    """高フレームレート（90fps）で左右カメラを表示"""
    print("高フレームレートステレオモード（90fps）...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 高フレームレート設定
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 90)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 90)
    
    try:
        pipeline.start(config)
        print("✓ 90fps ステレオストリーミング開始")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frames = pipeline.wait_for_frames()
            
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)
            
            if not left_frame or not right_frame:
                continue
            
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())
            
            stereo_image = np.hstack((left_image, right_image))
            
            # FPS計算
            frame_count += 1
            if frame_count % 90 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"実際のFPS: {fps:.1f}")
            
            cv2.putText(stereo_image, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(stereo_image, '90fps Mode', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            
            cv2.imshow('High FPS Stereo', stereo_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"エラー: {e}")
        
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== RealSense D435I 左右カメラ表示 ===")
    print("1: 基本的な左右カメラ表示")
    print("2: 全ストリーム表示（カラー+デプス+左右）")
    print("3: ステレオ視差解析")
    print("4: 高フレームレート（90fps）")
    print("何も入力しない場合は基本モードで実行")
    
    choice = input("選択 (1-4): ").strip()
    
    if choice == "2":
        show_all_streams()
    elif choice == "3":
        stereo_depth_analysis()
    elif choice == "4":
        high_fps_stereo()
    else:
        show_stereo_cameras()  # デフォルト