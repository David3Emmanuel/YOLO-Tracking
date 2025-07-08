from yolo_tracker import YoloTracker

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO tracking demo with embedding aggregation')
    parser.add_argument('--source', type=str, default="store.mp4", help='Source video file path')
    parser.add_argument('--skip-frames', type=int, default=5, help='Number of frames to skip')
    parser.add_argument('--output-path', type=str, default="output/results", help='Output directory path')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--save-images', action='store_true', help='Save images')
    parser.add_argument('--skip-consolidation', action='store_true', help='Skip consolidation')
    args = parser.parse_args()
    tracker = YoloTracker(
        source=args.source,
        skip_frames=args.skip_frames,
        output_path=args.output_path,
        preview=args.preview,
        save_video=args.save_video,
        save_images=args.save_images,
        skip_consolidation=args.skip_consolidation
    )
    try:
        tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.cleanup()