import type { Metadata } from "next";
import "@livekit/components-styles/prefabs/index.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Roxstar AI Voice Room Assistant",
  description: "Real-time AI Voice Room Assistant with Dost and Sathi",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
