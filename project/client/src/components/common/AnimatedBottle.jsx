import React, { useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';

export default function AnimatedBottle({ imageUrl, customDelay = 0, style = {} }) {
  const controls = useAnimation();

  useEffect(() => {
    let isMounted = true; // Flag to track mounting status

    async function runAnimationLoop() {
      while (isMounted) {
        try {
          // Step 1: Slide in from the left and fade in
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

          // Step 2: Teeter (rotate gently back and forth)
          await controls.start({
            rotate: [0, 5, -5, 0],
            transition: { duration: 2, ease: 'easeInOut' },
          });
          if (!isMounted) break;

          // Step 3: Float out to the right while fading out
          await controls.start({
            x: '100vw',
            opacity: 0,
            transition: { duration: 1.5, ease: 'easeInOut' },
          });
          if (!isMounted) break;

          // Reset the bottle's position so it can loop again
          controls.set({ x: '-100vw', opacity: 0, rotate: 0 });
          if (!isMounted) break;

          // Optional: wait a moment before repeating this bottle’s animation cycle
          await new Promise((resolve) => setTimeout(resolve, 1000));
        } catch (err) {
          console.error(err);
          break; // Exit the loop on error.
        }
      }
    }
    runAnimationLoop();

    // Cleanup function sets the flag to false to stop the loop
    return () => {
      isMounted = false;
    };
  }, [controls, customDelay]);

  return (
    <motion.div
      initial={{ x: '-100vw', opacity: 0, rotate: 0 }}
      animate={controls}
      style={{
        position: 'absolute',
        top: '180px',         // Adjust vertical positioning as needed
        left: '42%',          // This positions it near the red box; adjust as necessary
        transform: 'translateX(-42%)',
        width: '300px',       // Adjust width if needed
        height: '600px',      // Adjust height if needed
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
