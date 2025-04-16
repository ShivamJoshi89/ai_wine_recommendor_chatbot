import React, { useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';

export default function AnimatedBottle({ imageUrl, customDelay = 0, style = {} }) {
  const controls = useAnimation();

  useEffect(() => {
    let isMounted = true;

    // Before starting the loop, ensure initial position is set via the controls API
    // (this runs only once, after mount)
    controls.set({ x: '-100vw', opacity: 0, rotate: 0 });

    async function runAnimationLoop() {
      while (isMounted) {
        try {
          // 1) Slide in from left and fade in
          await controls.start({
            x: 0,
            opacity: 1,
            transition: {
              type: 'spring',
              stiffness: 50,
              damping: 20,
              duration: 1.5,
              delay: customDelay,
            },
          });
          if (!isMounted) break;

          // 2) Teeter
          await controls.start({
            rotate: [0, 5, -5, 0],
            transition: { duration: 2, ease: 'easeInOut' },
          });
          if (!isMounted) break;

          // 3) Float out to right while fading out
          await controls.start({
            x: '100vw',
            opacity: 0,
            transition: { duration: 1.5, ease: 'easeInOut' },
          });
          if (!isMounted) break;

          // 4) Reset position instantly (use start with zero-duration)
          await controls.start({
            x: '-100vw',
            opacity: 0,
            rotate: 0,
            transition: { duration: 0 },
          });
          if (!isMounted) break;

          // 5) Pause before next loop
          await new Promise((res) => setTimeout(res, 1000));
        } catch (err) {
          console.error(err);
          break;
        }
      }
    }

    runAnimationLoop();

    return () => {
      isMounted = false;
    };
  }, [controls, customDelay]);

  return (
    <motion.div
      animate={controls}
      style={{
        position: 'absolute',
        top: '180px',
        left: '42%',
        transform: 'translateX(-42%)',
        width: '300px',
        height: '600px',
        backgroundImage: `url(${imageUrl})`,
        backgroundSize: 'contain',
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center bottom',
        pointerEvents: 'none',
        zIndex: 0,
        ...style,
      }}
    />
  );
}
